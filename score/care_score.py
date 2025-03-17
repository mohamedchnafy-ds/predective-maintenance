class AnomalyEvent:
    """Classe représentant un événement d'anomalie avec ses caractéristiques temporelles et prédictions."""

    def __init__(self, start_idx: int, end_idx: int, timestamps: List, predictions: List[int], ground_truth: List[int]):
        """
        Initialise un événement d'anomalie.

        Args:
            start_idx: Indice de début de l'événement
            end_idx: Indice de fin de l'événement
            timestamps: Liste des horodatages de l'événement
            predictions: Liste des prédictions (0 ou 1) pour chaque horodatage
            ground_truth: Liste des vérités terrain (0 ou 1) pour chaque horodatage
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.timestamps = timestamps
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.length = end_idx - start_idx + 1

    def is_detected(self) -> bool:
        """Vérifie si l'événement est détecté (au moins une prédiction positive)."""
        return 1 in self.predictions
    
class EventDetector:
    """Classe utilitaire pour détecter des événements d'anomalie dans une série temporelle."""

    @staticmethod
    def detect_events(df: pd.DataFrame, status_column: str = 'status_id',
                      normal_value: str = 'normal', predictions_column: str = 'anomaly',
                      time_column: str = 'time_stamp', min_gap: int = 3) -> List[AnomalyEvent]:
        """
        Détecte des événements d'anomalie dans une série temporelle.

        Args:
            df: DataFrame contenant les données
            status_column: Nom de la colonne contenant le statut (normal ou anormal)
            normal_value: Valeur indiquant un comportement normal
            predictions_column: Nom de la colonne contenant les prédictions d'anomalie
            time_column: Nom de la colonne contenant les horodatages
            min_gap: Écart minimal entre deux événements distincts

        Returns:
            Liste des événements d'anomalie détectés
        """
        # Convertir le statut en valeurs binaires (0 pour normal, 1 pour anormal)
        df_binary = df.copy()
        df_binary['status_binary'] = (df_binary[status_column] != normal_value).astype(int)

        # Identifier les séquences d'anomalies (statut anormal consécutif)
        status_changes = df_binary['status_binary'].diff().fillna(0).ne(0).cumsum()

        events = []
        current_event = None
        previous_end = -min_gap - 1

        # Parcourir chaque groupe d'indices avec le même statut
        for group_id, indices in df_binary.groupby(status_changes).groups.items():
            if len(indices) == 0:
                continue

            start_idx = indices[0]
            end_idx = indices[-1]

            # Ne traiter que les groupes avec des anomalies (statut anormal)
            if df_binary.loc[start_idx, 'status_binary'] == 1:
                # Si l'écart avec l'événement précédent est trop petit, fusionner les événements
                if start_idx - previous_end <= min_gap:
                    # Étendre l'événement actuel
                    if current_event is not None:
                        current_event.end_idx = end_idx
                        current_event.timestamps.extend(df_binary.loc[previous_end+1:end_idx, time_column].tolist())
                        current_event.predictions.extend(df_binary.loc[previous_end+1:end_idx, predictions_column].tolist())
                        current_event.ground_truth.extend(df_binary.loc[previous_end+1:end_idx, 'status_binary'].tolist())
                        current_event.length = current_event.end_idx - current_event.start_idx + 1
                else:
                    # Créer un nouvel événement
                    timestamps = df_binary.loc[start_idx:end_idx, time_column].tolist()
                    predictions = df_binary.loc[start_idx:end_idx, predictions_column].astype(int).tolist()
                    ground_truth = df_binary.loc[start_idx:end_idx, 'status_binary'].tolist()

                    current_event = AnomalyEvent(start_idx, end_idx, timestamps, predictions, ground_truth)
                    events.append(current_event)

                previous_end = end_idx

        return events
    
class Coverage:
    """
    Calcule le score de Coverage pour un modèle de détection d'anomalies.
    Le score de Coverage est basé sur le F-score et mesure la performance
    de classification sur des données contenant des anomalies.
    """

    def __init__(self, beta: float = 0.5):
        """
        Initialise la classe Coverage.

        Args:
            beta: Paramètre beta pour le calcul du F-score (valeur par défaut: 0.5)
                 Une valeur inférieure à 1 donne plus de poids à la précision qu'au rappel,
                 ce qui pénalise les faux positifs.
        """
        self.beta = beta

    def filter_prediction_time_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre le DataFrame pour ne conserver que les points de données avec des anomalies.

        Args:
            df: DataFrame contenant les données, incluant une colonne 'status_id'

        Returns:
            DataFrame filtré ne contenant que les lignes où 'status_id' est 'not normal'
        """
        filtered_df = df[df['status_id'] == 'not normal']
        return filtered_df

    def calculate_fbeta_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calcule le score F-beta.

        Args:
            ground_truth: Tableau de vérités terrain (0 pour normal, 1 pour anomalie)
            predictions: Tableau de prédictions (0 pour normal, 1 pour anomalie)

        Returns:
            Score F-beta calculé
        """
        tp = np.sum((ground_truth == 1) & (predictions == 1))
        fn = np.sum((ground_truth == 1) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Éviter la division par zéro
        denominator = (1 + self.beta**2) * tp + self.beta**2 * fn + fp
        if denominator == 0:
            return 0.0

        fbeta = (1 + self.beta**2) * tp / denominator
        return fbeta

    def calculate_coverage_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """
        Calcule le score de Coverage pour le modèle de détection d'anomalies.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies

        Returns:
            Score de Coverage calculé
        """
        # Filtrer pour ne conserver que les points avec des anomalies
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("Attention: Aucun point de données avec des anomalies trouvé pour le calcul du Coverage.")
            return 0.0

        # Extraire les vérités terrain et les prédictions
        ground_truth = np.ones(len(filtered_df))  # Tous les points sont des anomalies par définition
        predictions = anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int).values

        # Calculer le score F-beta
        coverage_score = self.calculate_fbeta_score(ground_truth, predictions)
        return coverage_score

    def visualize_coverage(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                         time_column: str = 'time_stamp') -> None:
        """
        Visualise le score de Coverage à travers le temps.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies
            time_column: Nom de la colonne contenant les horodatages
        """
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("Pas de données disponibles pour la visualisation du Coverage.")
            return

        # Fusionner les données filtrées avec les prédictions
        merged_data = filtered_df.copy()
        merged_data['prediction'] = anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int)

        # Calculer les TP, FN, FP, TN
        merged_data['TP'] = (merged_data['prediction'] == 1).astype(int)
        merged_data['FN'] = (merged_data['prediction'] == 0).astype(int)

        # Trier par horodatage
        merged_data.sort_values(time_column, inplace=True)

        # Créer le graphique
        plt.figure(figsize=(14, 6))

        # Tracer les TP et FN
        plt.scatter(merged_data[time_column], merged_data['TP'],
                  label='Anomalies détectées (TP)', color='green', alpha=0.7)
        plt.scatter(merged_data[time_column], merged_data['FN'],
                  label='Anomalies manquées (FN)', color='red', alpha=0.7)

        plt.title('Performance de Détection des Anomalies (Coverage)')
        plt.xlabel('Temps')
        plt.ylabel('Détection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Accuracy:
    """
    Calcule le score d'Accuracy pour un modèle de détection d'anomalies.
    Le score d'Accuracy mesure la performance sur des données ne contenant
    que des comportements normaux, en se concentrant sur les vrais négatifs.
    """

    def filter_prediction_time_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtre le DataFrame pour ne conserver que les points de données normaux.

        Args:
            df: DataFrame contenant les données, incluant une colonne 'status_id'

        Returns:
            DataFrame filtré ne contenant que les lignes où 'status_id' est 'normal'
        """
        filtered_df = df[df['status_id'] == 'normal']
        return filtered_df

    def calculate_acc_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calcule le score d'Accuracy.

        Args:
            ground_truth: Tableau de vérités terrain (0 pour normal, 1 pour anomalie)
            predictions: Tableau de prédictions (0 pour normal, 1 pour anomalie)

        Returns:
            Score d'Accuracy calculé
        """
        tn = np.sum((ground_truth == 0) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Éviter la division par zéro
        if tn + fp == 0:
            return 0.0

        acc_score = tn / (tn + fp)
        return acc_score

    def calculate_accuracy_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """
        Calcule le score d'Accuracy pour le modèle de détection d'anomalies.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies

        Returns:
            Score d'Accuracy calculé
        """
        # Filtrer pour ne conserver que les points normaux
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("Attention: Aucun point de données normal trouvé pour le calcul de l'Accuracy.")
            return 0.0

        # Extraire les vérités terrain et les prédictions
        ground_truth = np.zeros(len(filtered_df))  # Tous les points sont normaux par définition
        predictions = anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int).values

        # Calculer le score d'Accuracy
        accuracy_score = self.calculate_acc_score(ground_truth, predictions)
        return accuracy_score

    def visualize_accuracy(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                         time_column: str = 'time_stamp') -> None:
        """
        Visualise le score d'Accuracy à travers le temps.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies
            time_column: Nom de la colonne contenant les horodatages
        """
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("Pas de données disponibles pour la visualisation de l'Accuracy.")
            return

        # Fusionner les données filtrées avec les prédictions
        merged_data = filtered_df.copy()
        merged_data['prediction'] = anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int)

        # Calculer les TN et FP
        merged_data['TN'] = (merged_data['prediction'] == 0).astype(int)
        merged_data['FP'] = (merged_data['prediction'] == 1).astype(int)

        # Trier par horodatage
        merged_data.sort_values(time_column, inplace=True)

        # Créer le graphique
        plt.figure(figsize=(14, 6))

        # Tracer les TN et FP
        plt.scatter(merged_data[time_column], merged_data['TN'],
                  label='Comportements normaux correctement identifiés (TN)', color='blue', alpha=0.7)
        plt.scatter(merged_data[time_column], merged_data['FP'],
                  label='Fausses alarmes (FP)', color='orange', alpha=0.7)

        plt.title('Performance sur les Comportements Normaux (Accuracy)')
        plt.xlabel('Temps')
        plt.ylabel('Détection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Reliability:
    """
    Calcule le score de Reliability pour un modèle de détection d'anomalies.
    Le score de Reliability prend en compte les fausses alarmes sur une base
    d'événements en utilisant un algorithme de criticité.
    """

    def __init__(self, beta: float = 0.5, threshold: int = 72):
        """
        Initialise la classe Reliability.

        Args:
            beta: Paramètre beta pour le calcul du F-score (valeur par défaut: 0.5)
            threshold: Seuil de criticité pour la classification des événements (valeur par défaut: 72)
                      Ce seuil correspond à 12 heures d'anomalies consécutives (avec des mesures toutes les 10 minutes)
        """
        self.beta = beta
        self.threshold = threshold

    def _calculate_criticality(self, status_info: np.ndarray, predictions: np.ndarray) -> List[int]:
        """
        Calcule la criticité pour chaque point temporel.

        Args:
            status_info: Tableau représentant le statut du système (0 pour anormal, 1 pour normal)
            predictions: Tableau représentant les prédictions d'anomalies (0 pour normal, 1 pour anomalie)

        Returns:
            Liste des valeurs de criticité
        """
        # Appliquer l'algorithme de criticité comme défini dans l'image
        crit = [0]

        for i in range(1, len(status_info)):
            if status_info[i] == 0:  # Statut anormal
                if predictions[i] == 1:  # Anomalie prédite
                    crit.append(crit[i-1] + 1)  # Incrémenter la criticité
                else:  # Pas d'anomalie prédite
                    crit.append(max(crit[i-1] - 1, 0))  # Décrémenter la criticité, min 0
            else:  # Statut normal
                crit.append(crit[i-1])  # Maintenir la criticité

        return crit

    def _classify_events(self, criticality: List[int], df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifie les événements basés sur la criticité.

        Args:
            criticality: Liste des valeurs de criticité
            df: DataFrame original contenant les données

        Returns:
            DataFrame avec une colonne supplémentaire 'event_detected'
        """
        # Trouver la criticité maximale
        max_criticality = max(criticality) if criticality else 0

        # Classifier les événements (1 si la criticité max > seuil, 0 sinon)
        df_with_events = df.copy()
        df_with_events['event_detected'] = 1 if max_criticality > self.threshold else 0

        return df_with_events

    def _calculate_fbeta_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calcule le score F-beta.

        Args:
            ground_truth: Tableau de vérités terrain (0 pour normal, 1 pour anomalie)
            predictions: Tableau de prédictions (0 pour normal, 1 pour anomalie)

        Returns:
            Score F-beta calculé
        """
        tp = np.sum((ground_truth == 1) & (predictions == 1))
        fn = np.sum((ground_truth == 1) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Éviter la division par zéro
        denominator = (1 + self.beta**2) * tp + self.beta**2 * fn + fp
        if denominator == 0:
            return 0.0

        fbeta = (1 + self.beta**2) * tp / denominator
        return fbeta

    def calculate_reliability_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """
        Calcule le score de Reliability pour le modèle de détection d'anomalies.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies

        Returns:
            Score de Reliability calculé
        """
        # Convertir le statut en valeurs binaires (1 pour normal, 0 pour anormal)
        status_info = df['status_id'].map({'normal': 1, 'not normal': 0}).values

        # Obtenir les prédictions
        predictions = anomaly_results['anomaly'].astype(int).values

        # Calculer la criticité pour chaque point temporel
        criticality = self._calculate_criticality(status_info, predictions)

        # Classifier les événements basés sur la criticité
        df_with_events = self._classify_events(criticality, df)

        # Extraire les vérités terrain et les prédictions d'événements
        ground_truth_events = df['status_id'].map({'normal': 0, 'not normal': 1}).values
        event_predictions = df_with_events['event_detected'].values

        # Calculer le score F-beta pour les événements
        reliability_score = self._calculate_fbeta_score(ground_truth_events, event_predictions)

        return reliability_score

    def visualize_criticality(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                           time_column: str = 'time_stamp') -> None:
        """
        Visualise la criticité et les événements détectés.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies
            time_column: Nom de la colonne contenant les horodatages
        """
        # Convertir le statut en valeurs binaires (1 pour normal, 0 pour anormal)
        status_info = df['status_id'].map({'normal': 1, 'not normal': 0}).values

        # Obtenir les prédictions
        predictions = anomaly_results['anomaly'].astype(int).values

        # Calculer la criticité pour chaque point temporel
        criticality = self._calculate_criticality(status_info, predictions)

        # Fusionner les données avec la criticité
        merged_data = df.copy()
        merged_data['criticality'] = criticality
        merged_data['prediction'] = predictions
        merged_data['status_binary'] = status_info

        # Trier par horodatage
        merged_data.sort_values(time_column, inplace=True)

        # Créer le graphique
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Tracer la criticité
        ax1.plot(merged_data[time_column], merged_data['criticality'], color='purple', label='Criticité')
        ax1.axhline(y=self.threshold, color='r', linestyle='--', label=f'Seuil ({self.threshold})')
        ax1.set_title('Criticité au Fil du Temps')
        ax1.set_ylabel('Criticité')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Tracer les statuts et prédictions
        ax2.scatter(merged_data[time_column], merged_data['status_binary'],
                  label='Statut Réel (0=anormal, 1=normal)', color='blue', alpha=0.5)
        ax2.scatter(merged_data[time_column], merged_data['prediction'],
                  label='Prédiction (1=anomalie)', color='red', alpha=0.5)
        ax2.set_title('Statuts et Prédictions')
        ax2.set_xlabel('Temps')
        ax2.set_ylabel('État')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()






class Earliness:
    """
    Calcule le score d'Earliness pour un modèle de détection d'anomalies.
    Le score d'Earliness mesure la rapidité avec laquelle le modèle détecte
    les anomalies, en donnant plus de poids aux détections précoces.
    """

    def _calculate_weights(self, event_length: int) -> np.ndarray:
        """
        Calcule les poids pour un événement d'anomalie en fonction de sa longueur.

        Args:
            event_length: Longueur de l'événement d'anomalie

        Returns:
            Tableau des poids pour chaque point temporel de l'événement
        """
        weights = np.ones(event_length)

        # Appliquer la fonction de pondération linéaire décroissante
        # comme décrit dans l'image: poids décroissants de 1 à 0 sur toute la longueur
        weights = np.linspace(1.0, 0.0, event_length)

        return weights

    def _calculate_weighted_score(self, event: AnomalyEvent) -> float:
        """
        Calcule le score pondéré (WS) pour un événement d'anomalie.

        Args:
            event: Objet AnomalyEvent contenant les informations sur l'événement

        Returns:
            Score pondéré calculé
        """
        if not event.is_detected():
            return 0.0

        # Calculer les poids pour cet événement
        weights = self._calculate_weights(event.length)

        # Calculer le score pondéré
        weighted_score = np.sum(weights * np.array(event.predictions)) / np.sum(weights)

        return weighted_score

    def calculate_earliness_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """
        Calcule le score d'Earliness pour le modèle de détection d'anomalies.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies

        Returns:
            Score d'Earliness calculé
        """
        # Détecter les événements d'anomalie
        event_detector = EventDetector()
        anomaly_events = event_detector.detect_events(df, predictions_column='anomaly')

        if not anomaly_events:
            print("Attention: Aucun événement d'anomalie détecté pour le calcul d'Earliness.")
            return 0.0

        # Calculer le score pondéré pour chaque événement
        weighted_scores = [self._calculate_weighted_score(event) for event in anomaly_events]

        # Calculer le score d'Earliness global (moyenne des scores pondérés)
        earliness_score = np.mean(weighted_scores) if weighted_scores else 0.0

        return earliness_score

    def visualize_earliness(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                          time_column: str = 'time_stamp') -> None:
        """
        Visualise le score d'Earliness pour chaque événement d'anomalie.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies
            time_column: Nom de la colonne contenant les horodatages
        """
        # Détecter les événements d'anomalie
        event_detector = EventDetector()
        anomaly_events = event_detector.detect_events(df, predictions_column='anomaly')

        if not anomaly_events:
            print("Pas d'événements d'anomalie à visualiser.")
            return

        # Calculer les scores pondérés
        scores = [self._calculate_weighted_score(event) for event in anomaly_events]

        # Extraire les informations des événements pour la visualisation
        event_ids = list(range(len(anomaly_events)))
        event_lengths = [event.length for event in anomaly_events]

        # Créer un graphique à barres pour les scores d'Earliness par événement
        plt.figure(figsize=(12, 6))
        bars = plt.bar(event_ids, scores, alpha=0.7)

        # Ajouter les longueurs d'événement comme étiquettes
        for i, (bar, length) in enumerate(zip(bars, event_lengths)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'Longueur: {length}', ha='center', va='bottom', rotation=0, fontsize=8)

        plt.title('Score d\'Earliness par Événement d\'Anomalie')
        plt.xlabel('ID de l\'Événement')
        plt.ylabel('Score d\'Earliness (WS)')
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Visualiser un exemple d'événement avec ses poids
        if anomaly_events:
            example_event = max(anomaly_events, key=lambda e: e.length)  # Prendre l'événement le plus long
            weights = self._calculate_weights(example_event.length)

            # Créer un graphique pour les poids
            plt.figure(figsize=(10, 5))
            plt.plot(range(example_event.length), weights, 'b-', linewidth=2)
            plt.title('Fonction de Pondération pour un Événement d\'Anomalie')
            plt.xlabel('Position Temporelle dans l\'Événement')
            plt.ylabel('Poids')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

class CAREScore:
    """
    Calcule le score CARE pour un modèle de détection d'anomalies.
    Le score CARE combine les quatre métriques: Coverage, Accuracy, Reliability et Earliness.
    """

    def __init__(self, omega1: float = 1.0, omega2: float = 1.0, omega3: float = 1.0, omega4: float = 2.0):
        """
        Initialise la classe CAREScore.

        Args:
            omega1: Poids pour le score Coverage (valeur par défaut: 1.0)
            omega2: Poids pour le score Accuracy (valeur par défaut: 1.0)
            omega3: Poids pour le score Reliability (valeur par défaut: 1.0)
            omega4: Poids pour le score Earliness (valeur par défaut: 2.0)
        """
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.omega4 = omega4

        # Initialiser les métriques individuelles
        self.coverage = Coverage(beta=0.5)
        self.accuracy = Accuracy()
        self.reliability = Reliability(beta=0.5, threshold=72)
        self.earliness = Earliness()

    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """
        Calcule la moyenne pondérée des scores individuels.

        Args:
            scores: Dictionnaire des scores individuels

        Returns:
            Moyenne pondérée calculée
        """
        weights_sum = self.omega1 + self.omega2 + self.omega3 + self.omega4

        weighted_sum = (
            self.omega1 * scores['coverage'] +
            self.omega2 * scores['accuracy'] +
            self.omega3 * scores['reliability'] +
            self.omega4 * scores['earliness']
        )

        return weighted_sum / weights_sum

    def calculate_care_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule le score CARE global et les sous-scores.

        Args:
            df: DataFrame original contenant les données
            anomaly_results: DataFrame contenant les prédictions d'anomalies

        Returns:
            Dictionnaire contenant le score CARE et les sous-scores
        """
        # Calculer les scores individuels
        coverage_score = self.coverage.calculate_coverage_score(df, anomaly_results)
        accuracy_score = self.accuracy.calculate_accuracy_score(df, anomaly_results)
        reliability_score = self.reliability.calculate_reliability_score(df, anomaly_results)
        earliness_score = self.earliness.calculate_earliness_score(df, anomaly_results)

        scores = {
            'coverage': coverage_score,
            'accuracy': accuracy_score,
            'reliability': reliability_score,
            'earliness': earliness_score
        }

        # Vérifier s'il y a des anomalies détectées
        any_anomalies_detected = anomaly_results['anomaly'].sum() > 0

        # Calculer le score CARE final selon les conditions
        if not any_anomalies_detected:
            care_score = 0.0
        elif accuracy_score < 0.5:
            care_score = accuracy_score
        else:
            care_score = self._calculate_weighted_average(scores)

        # Ajouter le score CARE au dictionnaire des résultats
        scores['care'] = care_score

        return scores

    def visualize_care_score(self, scores: Dict[str, float]) -> None:
        """
        Visualise le score CARE et ses composantes.

        Args:
            scores: Dictionnaire contenant le score CARE et les sous-scores
        """
        # Créer des graphiques pour visualiser les scores
        labels = ['Coverage', 'Accuracy', 'Reliability', 'Earliness', 'CARE']
        values = [scores['coverage'], scores['accuracy'], scores['reliability'],
                scores['earliness'], scores['care']]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        # Graphique à barres des scores
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values, color=colors, alpha=0.8)

        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')

        plt.title('Score CARE et ses Composantes')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Graphique en radar
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        # Préparer les données pour le graphique en radar
        radar_labels = labels[:-1]  # Exclure CARE
        radar_values = values[:-1]  # Exclure CARE
        num_vars = len(radar_labels)

        # Calculer les angles pour chaque axe
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()

        # Fermer le polygone
        radar_values += [radar_values[0]]
        angles += [angles[0]]
        radar_labels += [radar_labels[0]]

        # Tracer le polygone
        ax.plot(angles, radar_values, 'o-', linewidth=2, color='#9b59b6')
        ax.fill(angles, radar_values, alpha=0.25, color='#9b59b6')

        # Configurer le graphique
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Composantes du Score CARE', y=1.1)

        plt.tight_layout()
        plt.show()

def main():
    """
    Fonction principale pour démontrer l'utilisation des métriques d'évaluation.
    """
    # Cette fonction servirait à démontrer l'utilisation des classes ci-dessus
    # sur un ensemble de données réel. Par exemple:

    # 1. Charger les données et les prédictions
    # df = pd.read_csv('path_to_data.csv')
    # anomaly_results = pd.read_csv('path_to_predictions.csv')

    # 2. Calculer les scores
    # care_calculator = CAREScore()
    # scores = care_calculator.calculate_care_score(df, anomaly_results)

    # 3. Afficher les résultats
    # print(f"Score CARE: {scores['care']:.4f}")
    # print(f"  Coverage: {scores['coverage']:.4f}")
    # print(f"  Accuracy: {scores['accuracy']:.4f}")
    # print(f"  Reliability: {scores['reliability']:.4f}")
    # print(f"  Earliness: {scores['earliness']:.4f}")

    # 4. Visualiser les scores
    # care_calculator.visualize_care_score(scores)

    pass


if __name__ == "__main__":
    main()