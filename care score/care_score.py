from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





class AnomalyEvent:
    """Represents a detected anomaly event with temporal characteristics and prediction results.
    
    Captures the complete lifecycle of an anomaly event including timestamps, 
    model predictions, and ground truth labels for evaluation purposes.
    """

    def __init__(self, start_idx: int, end_idx: int, timestamps: List, predictions: List[int], ground_truth: List[int]):
        """Initialize an anomaly event container.
        
        Args:
            start_idx: Start index of the event in the dataset (inclusive)
            end_idx: End index of the event in the dataset (inclusive)
            timestamps: Sequence of datetime values for the event window
            predictions: Model's anomaly predictions (0=normal, 1=anomaly) 
            ground_truth: Verified labels (0=normal, 1=anomaly) for accuracy assessment
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.timestamps = timestamps
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.length = end_idx - start_idx + 1  # Event duration in data points

    def is_detected(self) -> bool:
        """Determine if the event was detected by the model.
        
        Returns:
            True if at least one prediction flag is positive (1) within the event window
        """
        return 1 in self.predictions

class EventDetector:
    """Identifies contiguous anomaly events in time series data using status transitions and prediction coherence.
    
    Implements a stateful detection algorithm that:
    - Segments data into consecutive status groups
    - Merges adjacent anomalies within temporal tolerance
    - Constructs event objects with full temporal context
    """

    @staticmethod
    def detect_events(df: pd.DataFrame, status_column: str = 'status_id',
                      normal_value: str = 'normal', predictions_column: str = 'anomaly',
                      time_column: str = 'time_stamp', min_gap: int = 3) -> List[AnomalyEvent]:
        """Detects and clusters anomaly events in temporal data.
        
        Args:
            df: Input time series data with status and prediction columns
            status_column: Categorical column indicating system state
            normal_value: Reference value for normal operational status
            predictions_column: Binary model predictions (0=normal, 1=anomaly)
            time_column: DateTime column for temporal sequencing
            min_gap: Minimum allowable gap (in data points) between distinct events

        Returns:
            Chronologically ordered list of consolidated anomaly events
        """
        # Convert status to binary indicators (1=anomaly, 0=normal)
        df_binary = df.copy()
        df_binary['status_binary'] = (df_binary[status_column] != normal_value).astype(int)

        # Identify state transition points between normal/anomaly periods
        status_changes = df_binary['status_binary'].diff().fillna(0).ne(0).cumsum()

        events = []
        current_event = None
        previous_end = -min_gap - 1  # Initialize with impossible index

        # Track event state across consecutive groups
        for group_id, indices in df_binary.groupby(status_changes).groups.items():
            if not indices:
                continue

            start_idx, end_idx = indices[0], indices[-1]

            # Process only anomaly state groups
            if df_binary.loc[start_idx, 'status_binary'] == 1:
                # Merge events within temporal tolerance
                if start_idx - previous_end <= min_gap:
                    if current_event:
                        # Extend current event parameters
                        current_event.end_idx = end_idx
                        current_event.timestamps.extend(df_binary.loc[previous_end+1:end_idx, time_column].tolist())
                        current_event.predictions.extend(df_binary.loc[previous_end+1:end_idx, predictions_column].tolist())
                        current_event.ground_truth.extend(df_binary.loc[previous_end+1:end_idx, 'status_binary'].tolist())
                        current_event.length = current_event.end_idx - current_event.start_idx + 1
                else:
                    # Initialize new event with current group data
                    timestamps = df_binary.loc[start_idx:end_idx, time_column].tolist()
                    predictions = df_binary.loc[start_idx:end_idx, predictions_column].astype(int).tolist()
                    ground_truth = df_binary.loc[start_idx:end_idx, 'status_binary'].tolist()

                    current_event = AnomalyEvent(start_idx, end_idx, timestamps, predictions, ground_truth)
                    events.append(current_event)

                previous_end = end_idx

        return events

class Coverage:
    """Evaluates anomaly detection performance during anomalous periods using beta-adjusted F-score.
    
    Measures model's ability to:
    - Correctly identify true anomalies (Recall emphasis when beta > 1)
    - Maintain precision in anomaly claims (Precision emphasis when beta < 1)
    Optimized for operational scenarios where false negatives during anomalies are critical.
    """

    def __init__(self, beta: float = 0.5):
        """Configures the trade-off between precision and recall for anomaly detection.
        
        Args:
            beta: Weighting factor for recall in F-score calculation (default: 0.5)
                  Values < 1 prioritize precision (reduce false alarms)
                  Values > 1 prioritize recall (reduce missed anomalies)
        """
        self.beta = beta

    def filter_prediction_time_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolates anomalous periods for focused performance evaluation.
        
        Args:
            df: Raw dataset containing 'status_id' column with operational states

        Returns:
            Subset DataFrame containing only entries marked as 'not normal'
        """
        filtered_df = df[df['status_id'] == 'not normal']
        return filtered_df

    def calculate_fbeta_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Computes beta-weighted harmonic mean of precision and recall.
        
        Args:
            ground_truth: Binary labels (0=normal, 1=verified anomaly)
            predictions: Model outputs (0=normal, 1=predicted anomaly)

        Returns:
            F-beta score between 0 (worst) and 1 (perfect detection)
        """
        tp = np.sum((ground_truth == 1) & (predictions == 1))
        fn = np.sum((ground_truth == 1) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Safeguard against undefined scores
        denominator = (1 + self.beta**2) * tp + self.beta**2 * fn + fp
        return (1 + self.beta**2) * tp / denominator if denominator else 0.0

    def calculate_coverage_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """Calculates anomaly detection effectiveness during known problem periods.
        
        Args:
            df: Complete dataset with ground truth statuses
            anomaly_results: Model predictions with matching temporal indices

        Returns:
            Coverage score quantifying detection performance during anomalies
        """
        filtered_df = self.filter_prediction_time_frame(df)
        
        if filtered_df.empty:
            print("Warning: No anomalous periods found for coverage calculation.")
            return 0.0
            
        # Align prediction data with filtered anomalies
        anomaly_results = anomaly_results.reindex(filtered_df.index)
        
        # All filtered points are confirmed anomalies by definition
        ground_truth = np.ones(len(filtered_df))
        predictions = anomaly_results['anomaly'].astype(int).values

        return self.calculate_fbeta_score(ground_truth, predictions)

    def visualize_coverage(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                         time_column: str = 'time_stamp') -> None:
        """Generates temporal visualization of detection performance during anomalies.
        
        Args:
            df: Complete dataset with temporal index
            anomaly_results: Model predictions to visualize
            time_column: Temporal coordinate for x-axis visualization
        """
        filtered_df = self.filter_prediction_time_frame(df)
        
        if filtered_df.empty:
            print("No anomalous data available for visualization.")
            return

        # Prepare visualization dataset
        viz_data = filtered_df.assign(
            prediction=anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int)
        ).sort_values(time_column)
        
        # Calculate performance metrics
        viz_data['TP'] = (viz_data['prediction'] == 1).astype(int)
        viz_data['FN'] = (viz_data['prediction'] == 0).astype(int)

        # Create annotated plot
        plt.figure(figsize=(14, 6))
        plt.scatter(viz_data[time_column], viz_data['TP'], 
                   label='True Anomalies Detected', color='#2ecc71', alpha=0.7, marker='^')
        plt.scatter(viz_data[time_column], viz_data['FN'],
                   label='Undetected Anomalies', color='#e74c3c', alpha=0.7, marker='x')
        
        plt.title(f'Anomaly Coverage Performance (β={self.beta})')
        plt.xlabel('Timestamp')
        plt.ylabel('Detection Outcome')
        plt.legend()
        plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.show()

class Accuracy:
    """Evaluates model performance during normal operation periods by measuring true negative dominance.
    
    Focuses on minimizing false positives in normal conditions through:
    - Strict monitoring of operational stability periods
    - Precision-focused assessment of negative predictions
    - Temporal visualization of prediction trustworthiness
    """

    def filter_prediction_time_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolates normal operational periods for specificity analysis.
        
        Args:
            df: Raw dataset containing 'status_id' column with system states

        Returns:
            Subset DataFrame containing only entries marked as 'normal'
        """
        filtered_df = df[df['status_id'] == 'normal']
        return filtered_df

    def calculate_acc_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Computes specificity score for normal operation periods.
        
        Args:
            ground_truth: Verified labels (0=normal, 1=anomaly)
            predictions: Model outputs (0=normal, 1=predicted anomaly)

        Returns:
            Specificity score between 0 (all FP) and 1 (all TN)
        """
        tn = np.sum((ground_truth == 0) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Handle edge case with no normal predictions
        return tn / (tn + fp) if (tn + fp) else 0.0

    def calculate_accuracy_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """Assesses model behavior during verified normal operation windows.
        
        Args:
            df: Complete dataset with ground truth statuses
            anomaly_results: Model predictions with temporal indices

        Returns:
            Accuracy metric quantifying reliable normal operation recognition
        """
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("Warning: No normal operation periods found for accuracy calculation.")
            return 0.0

        # All filtered points are confirmed normal by definition
        ground_truth = np.zeros(len(filtered_df))
        predictions = anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int).values

        return self.calculate_acc_score(ground_truth, predictions)

    def visualize_accuracy(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                         time_column: str = 'time_stamp') -> None:
        """Generates temporal view of normal operation prediction reliability.
        
        Args:
            df: Complete dataset with temporal index
            anomaly_results: Model predictions to visualize
            time_column: DateTime column for x-axis representation
        """
        filtered_df = self.filter_prediction_time_frame(df)

        if filtered_df.empty:
            print("No normal operation data available for visualization.")
            return

        # Prepare annotated visualization dataset
        viz_data = filtered_df.assign(
            prediction=anomaly_results.loc[filtered_df.index, 'anomaly'].astype(int)
        ).sort_values(time_column)

        # Calculate diagnostic metrics
        viz_data['TN'] = (viz_data['prediction'] == 0).astype(int)
        viz_data['FP'] = (viz_data['prediction'] == 1).astype(int)

        # Create comparative timeline plot
        plt.figure(figsize=(14, 6))
        plt.scatter(viz_data[time_column], viz_data['TN'],
                  label='True Negatives (Normal Behavior)', 
                  color='#3498db', alpha=0.7, marker='o')
        plt.scatter(viz_data[time_column], viz_data['FP'],
                  label='False Positives (Unwarranted Alerts)', 
                  color='#e67e22', alpha=0.7, marker='x')

        plt.title('Normal Operation Prediction Fidelity')
        plt.xlabel('Timestamp')
        plt.ylabel('Prediction Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class Reliability:
    """Evaluates model reliability by analyzing false alarms through event criticality analysis.
    
    Measures operational trustworthiness by:
    - Tracking sustained anomaly prediction confidence (criticality accumulation)
    - Classifying events based on criticality thresholds
    - Balancing detection accuracy with alert fatigue prevention
    """

    def __init__(self, beta: float = 0.5, threshold: int = 72):
        """Configures reliability assessment parameters for operational environments.
        
        Args:
            beta: F-score weighting factor emphasizing precision (β < 1) or recall (β > 1)
            threshold: Criticality threshold (in data points) for significant event classification
                      Default 72 points = 12 hours at 10-minute intervals
        """
        self.beta = beta
        self.threshold = threshold

    def _calculate_criticality(self, status_info: np.ndarray, predictions: np.ndarray) -> List[int]:
        """Computes temporal criticality scores through stateful anomaly confirmation.
        
        Args:
            status_info: Verified system states (0=abnormal, 1=normal)
            predictions: Model outputs (0=normal, 1=predicted anomaly)

        Returns:
            Criticality progression tracking sustained anomaly confidence
        """
        # Stateful algorithm accumulating prediction confidence during anomalies
        crit = [0]

        for i in range(1, len(status_info)):
            if status_info[i] == 0:  # Verified anomaly period
                if predictions[i] == 1:  # Correct prediction
                    crit.append(crit[i-1] + 1)  # Increase confidence
                else:  # Missed detection
                    crit.append(max(crit[i-1] - 1, 0))  # Decrease confidence
            else:  # Normal operation
                crit.append(crit[i-1])  # Maintain current level

        return crit

    def _classify_events(self, criticality: List[int], df: pd.DataFrame) -> pd.DataFrame:
        """Identifies significant events exceeding criticality thresholds.
        
        Args:
            criticality: Temporal criticality progression
            df: Source dataset for event annotation

        Returns:
            Dataset augmented with event classification flags
        """
        max_criticality = max(criticality) if criticality else 0
        
        # Flag events surpassing operational reliability thresholds
        df_with_events = df.copy()
        df_with_events['event_detected'] = int(max_criticality > self.threshold)
        
        return df_with_events

    def _calculate_fbeta_score(self, ground_truth: np.ndarray, predictions: np.ndarray) -> float:
        """Computes event-level F-beta score for reliability assessment.
        
        Args:
            ground_truth: Verified event labels (0=normal, 1=anomaly)
            predictions: Model event classifications (0=non-event, 1=event)

        Returns:
            Reliability score between 0 (unreliable) and 1 (perfect event detection)
        """
        tp = np.sum((ground_truth == 1) & (predictions == 1))
        fn = np.sum((ground_truth == 1) & (predictions == 0))
        fp = np.sum((ground_truth == 0) & (predictions == 1))

        # Safeguard against undefined scores
        denominator = (1 + self.beta**2) * tp + self.beta**2 * fn + fp
        return (1 + self.beta**2) * tp / denominator if denominator else 0.0

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
    """Evaluates detection timeliness by prioritizing early anomaly identification during events.
    
    Measures model's proactive detection capability through:
    - Time-decaying weights emphasizing early event phase detections
    - Weighted scoring of detection patterns
    - Visual analytics of detection timelines
    """

    def _calculate_weights(self, event_length: int) -> np.ndarray:
        """Generates temporal weighting profile for anomaly event evaluation.
        
        Args:
            event_length: Duration of continuous anomaly event (data points)

        Returns:
            Linear weight decay array from 1.0 (start) to 0.0 (end)
        """
        weights = np.ones(event_length)

        # Linear decay pattern incentivizes early detection
        weights = np.linspace(1.0, 0.0, event_length)

        return weights

    def _calculate_weighted_score(self, event: AnomalyEvent) -> float:
        """Computes detection timeliness score for individual anomaly events.
        
        Args:
            event: Temporal anomaly event container with prediction sequence

        Returns:
            Normalized score between 0 (late/no detection) and 1 (immediate detection)
        """
        if not event.is_detected():
            return 0.0
        
        # Event-length-adaptive weighting profile
        weights = self._calculate_weights(len(event.predictions))

        # Time-weighted detection performance calculation
        weighted_score = np.sum(weights * np.array(event.predictions)) / np.sum(weights)

        return weighted_score

    def calculate_earliness_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> float:
        """Calculates composite timeliness score across all detected anomaly events.
        
        Args:
            df: Ground truth dataset with system status
            anomaly_results: Model predictions aligned with temporal indices

        Returns:
            Global earliness metric (mean of individual event scores)
        """
        # Event detection and scoring pipeline
        event_detector = EventDetector()
        anomaly_events = event_detector.detect_events(df, predictions_column='anomaly')

        if not anomaly_events:
            print("Operational note: No anomaly events detected for timeliness evaluation.")
            return 0.0

        weighted_scores = [self._calculate_weighted_score(event) for event in anomaly_events]
        earliness_score = np.mean(weighted_scores) if weighted_scores else 0.0

        return earliness_score

    def visualize_earliness(self, df: pd.DataFrame, anomaly_results: pd.DataFrame,
                          time_column: str = 'time_stamp') -> None:
        """Generates diagnostic visualizations of detection timeliness patterns.
        
        Args:
            df: Source dataset with temporal index
            anomaly_results: Prediction data for visualization
            time_column: Temporal coordinate for event alignment
        """
        # Event detection and processing
        event_detector = EventDetector()
        anomaly_events = event_detector.detect_events(df, predictions_column='anomaly')

        if not anomaly_events:
            print("Visualization aborted: No anomaly events available for timeliness analysis.")
            return

        # Score calculation and visualization data preparation
        scores = [self._calculate_weighted_score(event) for event in anomaly_events]
        event_ids = list(range(len(anomaly_events)))
        event_lengths = [event.length for event in anomaly_events]

        # Composite visualization layout
        plt.figure(figsize=(12, 6))
        bars = plt.bar(event_ids, scores, alpha=0.7)

        # Annotate event characteristics
        for i, (bar, length) in enumerate(zip(bars, event_lengths)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'Duration: {length}', ha='center', va='bottom', fontsize=8)

        plt.title('Anomaly Detection Timeliness by Event')
        plt.xlabel('Event ID')
        plt.ylabel('Weighted Detection Score')
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


class CAREScore:
    """Composite evaluation metric combining four detection quality dimensions:
    - Coverage: Anomaly detection completeness
    - Accuracy: Normal operation specificity  
    - Reliability: Alert confidence sustainability
    - Earliness: Critical anomaly anticipation
    
    Provides weighted aggregation with configurable operational priorities.
    """

    def __init__(self, omega1: float = 1.0, omega2: float = 1.0, omega3: float = 1.0, omega4: float = 2.0):
        """Configures CARE score calculation parameters for industrial monitoring systems.
        
        Args:
            omega1: Coverage weight - detection completeness emphasis (default: 1.0)
            omega2: Accuracy weight - false positive avoidance emphasis (default: 1.0)
            omega3: Reliability weight - sustained detection confidence (default: 1.0)
            omega4: Earliness weight - early detection prioritization (default: 2.0)
        """
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.omega4 = omega4

        # Initialize metric calculators with industrial default configurations
        self.coverage = Coverage(beta=0.5)  # Balanced F0.5-score
        self.accuracy = Accuracy()  # Specificity-focused
        self.reliability = Reliability(beta=0.5, threshold=72)  # 12-hour criticality
        self.earliness = Earliness()  # Linear decay weighting

    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """Computes normalized weighted sum of component scores.
        
        Args:
            scores: Dictionary containing individual metric scores
            
        Returns:
            Normalized composite score between 0.0 (worst) and 1.0 (best)
            
        Formula:
            Σ(ω_i * score_i) / Σω_i  // Weighted average preserving [0,1] range
        """
        weights_sum = self.omega1 + self.omega2 + self.omega3 + self.omega4
        weighted_sum = (self.omega1 * scores['coverage'] +
                      self.omega2 * scores['accuracy'] +
                      self.omega3 * scores['reliability'] +
                      self.omega4 * scores['earliness'])
        return weighted_sum / weights_sum

    def calculate_care_score(self, df: pd.DataFrame, anomaly_results: pd.DataFrame) -> Dict[str, float]:
        """Computes comprehensive quality assessment for anomaly detection systems.
        
        Args:
            df: Ground truth dataset with temporal system status
            anomaly_results: Model predictions with temporal alignment
            
        Returns:
            Dictionary containing:
            - Individual metric scores (0.0-1.0)
            - Final CARE score incorporating operational logic and fallbacks
        """
        # Component score calculation pipeline
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

        # Operational logic gates
        any_anomalies_detected = anomaly_results['anomaly'].sum() > 0
        if not any_anomalies_detected:
            care_score = 0.0  # No detection scenario
        elif accuracy_score < 0.5:
            care_score = accuracy_score  # Accuracy critical failure
        else:
            care_score = self._calculate_weighted_average(scores)  # Normal operation

        scores['care'] = care_score
        return scores

    def visualize_care_score(self, scores: Dict[str, float]) -> None:
        """Generates dual visualization of composite score components:
        1. Bar chart: Individual metric contributions and final score
        2. Radar plot: Metric balance and performance profile
        
        Args:
            scores: Dictionary containing CARE score components
        """
        # Visualization data preparation
        labels = ['Coverage', 'Accuracy', 'Reliability', 'Earliness', 'CARE']
        values = [scores['coverage'], scores['accuracy'], 
                scores['reliability'], scores['earliness'], scores['care']]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        # Comparative bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values, color=colors, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        plt.title('CARE Score Component Breakdown')
        plt.ylabel('Performance Metric')
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Radar plot for metric balance
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        radar_labels = labels[:-1]
        radar_values = values[:-1]
        angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
        radar_values += [radar_values[0]]
        angles += [angles[0]]
        ax.plot(angles, radar_values, 'o-', linewidth=2, color='#9b59b6')
        ax.fill(angles, radar_values, alpha=0.25, color='#9b59b6')
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Metric Balance Radar', y=1.1)
        plt.tight_layout()
        plt.show()



