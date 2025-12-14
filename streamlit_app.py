"""
Interface Streamlit pour tester le modèle de détection DDoS
===========================================================
Cette application permet de :
- Simuler des attaques en utilisant des échantillons du dataset
- Tester la détection du modèle
- Analyser les faux positifs et faux négatifs
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import glob
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import time

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="DDoS Detection Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    """Charge le modèle et les artefacts nécessaires."""
    try:
        # Charger le modèle
        model_path = 'ddos_lstm_multiclass_MERGED_final.keras'
        if not Path(model_path).exists():
            model_path = 'best_lstm_ddos_model.keras'
        
        if not Path(model_path).exists():
            return None, None, None, None, "Modèle non trouvé. Veuillez d'abord entraîner le modèle."
        
        model = tf.keras.models.load_model(model_path)
        
        # Charger le scaler
        scaler = joblib.load('scaler_merged.pkl')
        
        # Charger le label encoder
        label_encoder = joblib.load('label_encoder_merged.pkl')
        
        # Charger les paramètres
        try:
            params = joblib.load('model_params_merged.pkl')
            sequence_length = params.get('sequence_length', 10)
            step = params.get('step', 5)
        except:
            sequence_length = 10
            step = 5
            params = None
        
        return model, scaler, label_encoder, params, None
    except Exception as e:
        return None, None, None, None, f"Erreur lors du chargement: {str(e)}"

@st.cache_data
def load_dataset_samples():
    """Charge des échantillons de chaque type d'attaque depuis le dataset."""
    data_path = "./ddos_csv_files/"
    samples = {}
    
    # Charger tous les fichiers CSV
    csv_files = glob.glob(f'{data_path}*.csv')
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'Label' in df.columns:
                # Prendre quelques échantillons de chaque label dans ce fichier
                for label in df['Label'].unique():
                    label_samples = df[df['Label'] == label].head(100)  # Max 100 échantillons par label
                    if label not in samples:
                        samples[label] = []
                    samples[label].append(label_samples)
        except Exception as e:
            continue
    
    # Fusionner les échantillons par label
    final_samples = {}
    for label, dfs in samples.items():
        if dfs:
            final_samples[label] = pd.concat(dfs, ignore_index=True).drop_duplicates()
    
    return final_samples

def create_sequences(X, seq_length=10, step=5):
    """Crée des séquences temporelles pour le modèle LSTM."""
    if len(X) < seq_length:
        # Si pas assez de données, répéter les données
        X = np.tile(X, (seq_length // len(X) + 1, 1))[:seq_length]
        return np.array([X]), [0]
    
    X_seq = []
    indices = []
    
    for i in range(0, len(X) - seq_length + 1, step):
        X_seq.append(X[i:i+seq_length])
        indices.append(i)
    
    if len(X_seq) == 0:
        # Si toujours pas de séquences, créer une seule séquence avec les dernières données
        X_seq.append(X[-seq_length:])
        indices.append(max(0, len(X) - seq_length))
    
    return np.array(X_seq), indices

def predict_sequences(model, scaler, label_encoder, data, sequence_length=10, step=5):
    """Fait des prédictions sur les données."""
    # Séparer features et labels
    if 'Label' in data.columns:
        X = data.drop('Label', axis=1)
        y_true = data['Label'].values
    else:
        X = data
        y_true = None
    
    # Normaliser
    X_scaled = scaler.transform(X)
    
    # Créer des séquences
    X_seq, indices = create_sequences(X_scaled, sequence_length, step)
    
    if len(X_seq) == 0:
        return None, None, None, None
    
    # Prédictions
    y_pred_proba = model.predict(X_seq, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Labels réels pour les séquences
    if y_true is not None:
        y_true_seq = [y_true[i+sequence_length-1] for i in indices]
    else:
        y_true_seq = None
    
    return y_pred_labels, y_pred_proba, y_true_seq, indices

def main():
    # Header
    st.markdown('<div class="main-header">🛡️ DDoS Detection Simulator</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Charger le modèle
    with st.sidebar:
        st.subheader("📦 Chargement du modèle")
        if st.button("🔄 Charger le modèle", type="primary"):
            with st.spinner("Chargement du modèle..."):
                model, scaler, label_encoder, params, error = load_model_and_artifacts()
                if error:
                    st.error(error)
                    st.stop()
                else:
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['label_encoder'] = label_encoder
                    st.session_state['params'] = params
                    st.success("✅ Modèle chargé avec succès!")
    
    # Vérifier si le modèle est chargé
    if 'model' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger le modèle depuis la sidebar.")
        st.info("💡 Le modèle doit être entraîné et sauvegardé avant d'utiliser cette interface.")
        st.stop()
    
    model = st.session_state['model']
    scaler = st.session_state['scaler']
    label_encoder = st.session_state['label_encoder']
    params = st.session_state.get('params', {})
    sequence_length = params.get('sequence_length', 10) if params else 10
    step = params.get('step', 5) if params else 5
    
    # Informations sur le modèle
    with st.expander("ℹ️ Informations sur le modèle"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classes", len(label_encoder.classes_))
        with col2:
            st.metric("Sequence Length", sequence_length)
        with col3:
            st.metric("Step", step)
        with col4:
            st.metric("Features", model.input_shape[2])
        st.write("**Classes détectables:**", ", ".join(label_encoder.classes_))
    
    # Charger les échantillons du dataset
    with st.spinner("Chargement des échantillons du dataset..."):
        dataset_samples = load_dataset_samples()
    
    if not dataset_samples:
        st.error("❌ Aucun échantillon trouvé dans le dataset. Vérifiez que les fichiers CSV sont dans 'ddos_csv_files/'")
        st.stop()
    
    # Interface principale
    st.header("🎯 Simulation d'attaques")
    
    # Sélection du type de test
    test_type = st.radio(
        "Type de test:",
        ["🔴 Simuler une attaque spécifique", "🟢 Tester le trafic bénin", "📊 Test complet (toutes les classes)"],
        horizontal=True
    )
    
    if test_type == "🔴 Simuler une attaque spécifique":
        # Sélection de l'attaque
        available_labels = [label for label in dataset_samples.keys() if label != 'Benign']
        selected_attack = st.selectbox(
            "Sélectionner le type d'attaque à simuler:",
            available_labels,
            help="Choisissez un type d'attaque DDoS à tester"
        )
        
        # Nombre d'échantillons
        n_samples = st.slider(
            "Nombre d'échantillons à tester:",
            min_value=10,
            max_value=min(1000, len(dataset_samples[selected_attack])),
            value=100,
            step=10
        )
        
        if st.button("🚀 Lancer la simulation", type="primary"):
            with st.spinner(f"Simulation de {n_samples} échantillons de {selected_attack}..."):
                # Prendre les échantillons
                test_data = dataset_samples[selected_attack].head(n_samples).copy()
                
                # Prédictions
                y_pred, y_proba, y_true, indices = predict_sequences(
                    model, scaler, label_encoder, test_data, sequence_length, step
                )
                
                if y_pred is not None:
                    st.session_state['results'] = {
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'y_true': y_true,
                        'true_label': selected_attack,
                        'test_type': 'attack'
                    }
                    st.rerun()
    
    elif test_type == "🟢 Tester le trafic bénin":
        if 'Benign' not in dataset_samples:
            st.warning("⚠️ Aucun échantillon bénin trouvé dans le dataset.")
        else:
            # Nombre d'échantillons
            n_samples = st.slider(
                "Nombre d'échantillons bénins à tester:",
                min_value=10,
                max_value=min(1000, len(dataset_samples['Benign'])),
                value=100,
                step=10
            )
            
            if st.button("🚀 Tester le trafic bénin", type="primary"):
                with st.spinner(f"Test de {n_samples} échantillons bénins..."):
                    # Prendre les échantillons
                    test_data = dataset_samples['Benign'].head(n_samples).copy()
                    
                    # Prédictions
                    y_pred, y_proba, y_true, indices = predict_sequences(
                        model, scaler, label_encoder, test_data, sequence_length, step
                    )
                    
                    if y_pred is not None:
                        st.session_state['results'] = {
                            'y_pred': y_pred,
                            'y_proba': y_proba,
                            'y_true': y_true,
                            'true_label': 'Benign',
                            'test_type': 'benign'
                        }
                        st.rerun()
    
    elif test_type == "📊 Test complet (toutes les classes)":
        n_samples_per_class = st.slider(
            "Nombre d'échantillons par classe:",
            min_value=5,
            max_value=100,
            value=50,
            step=5
        )
        
        if st.button("🚀 Lancer le test complet", type="primary"):
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (label, data) in enumerate(dataset_samples.items()):
                status_text.text(f"Test de {label}... ({idx+1}/{len(dataset_samples)})")
                progress_bar.progress((idx + 1) / len(dataset_samples))
                
                test_data = data.head(n_samples_per_class).copy()
                y_pred, y_proba, y_true, indices = predict_sequences(
                    model, scaler, label_encoder, test_data, sequence_length, step
                )
                
                if y_pred is not None:
                    for pred, true_lbl in zip(y_pred, y_true if y_true else [label]*len(y_pred)):
                        all_results.append({
                            'True Label': true_lbl if true_lbl else label,
                            'Predicted Label': pred
                        })
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                st.session_state['results'] = {
                    'results_df': results_df,
                    'test_type': 'complete'
                }
                st.rerun()
    
    # Afficher les résultats
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        st.markdown("---")
        st.header("📊 Résultats de la détection")
        
        if results['test_type'] == 'complete':
            # Test complet
            results_df = results['results_df']
            
            # Matrice de confusion
            st.subheader("📈 Matrice de confusion")
            cm = confusion_matrix(results_df['True Label'], results_df['Predicted Label'],
                                 labels=label_encoder.classes_)
            
            # Visualisation avec Plotly
            fig = px.imshow(
                cm,
                labels=dict(x="Prédit", y="Réel", color="Nombre"),
                x=label_encoder.classes_,
                y=label_encoder.classes_,
                aspect="auto",
                color_continuous_scale="Blues"
            )
            fig.update_layout(title="Matrice de confusion - Toutes les classes")
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques par classe
            st.subheader("📋 Métriques par classe")
            report = classification_report(
                results_df['True Label'],
                results_df['Predicted Label'],
                target_names=label_encoder.classes_,
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Statistiques globales
            col1, col2, col3, col4 = st.columns(4)
            accuracy = (results_df['True Label'] == results_df['Predicted Label']).mean()
            with col1:
                st.metric("Accuracy globale", f"{accuracy:.2%}")
            
            # Faux positifs et faux négatifs
            st.subheader("🔍 Analyse des erreurs")
            
            # Faux positifs (prédit attaque mais réellement bénin)
            fp = results_df[(results_df['True Label'] == 'Benign') & 
                           (results_df['Predicted Label'] != 'Benign')]
            fp_rate = len(fp) / len(results_df[results_df['True Label'] == 'Benign']) * 100 if len(results_df[results_df['True Label'] == 'Benign']) > 0 else 0
            
            # Faux négatifs (prédit bénin mais réellement attaque)
            fn = results_df[(results_df['True Label'] != 'Benign') & 
                           (results_df['Predicted Label'] == 'Benign')]
            fn_rate = len(fn) / len(results_df[results_df['True Label'] != 'Benign']) * 100 if len(results_df[results_df['True Label'] != 'Benign']) > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faux positifs", f"{len(fp)} ({fp_rate:.2f}%)")
                if len(fp) > 0:
                    st.dataframe(fp.head(10), use_container_width=True)
            
            with col2:
                st.metric("Faux négatifs", f"{len(fn)} ({fn_rate:.2f}%)")
                if len(fn) > 0:
                    st.dataframe(fn.head(10), use_container_width=True)
        
        else:
            # Test d'une classe spécifique
            y_pred = results['y_pred']
            y_proba = results['y_proba']
            y_true = results['y_true']
            true_label = results['true_label']
            
            # Statistiques de détection
            col1, col2, col3, col4 = st.columns(4)
            
            if results['test_type'] == 'attack':
                # Pour les attaques
                correct = (y_pred == true_label).sum()
                accuracy = correct / len(y_pred) * 100
                
                with col1:
                    st.metric("Précision", f"{accuracy:.2f}%")
                with col2:
                    st.metric("Détections correctes", f"{correct}/{len(y_pred)}")
                with col3:
                    st.metric("Faux négatifs", f"{(y_pred != true_label).sum()}")
                with col4:
                    st.metric("Confiance moyenne", f"{y_proba.max(axis=1).mean():.2%}")
                
                # Distribution des prédictions
                st.subheader("📊 Distribution des prédictions")
                pred_counts = pd.Series(y_pred).value_counts()
                fig = px.bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    labels={'x': 'Classe prédite', 'y': 'Nombre'},
                    title=f"Prédictions pour {true_label}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Probabilités
                st.subheader("🎯 Probabilités de prédiction")
                proba_df = pd.DataFrame(y_proba, columns=label_encoder.classes_)
                proba_df['Predicted'] = y_pred
                proba_df['Correct'] = (y_pred == true_label)
                
                # Graphique des probabilités
                fig = go.Figure()
                for i, class_name in enumerate(label_encoder.classes_):
                    fig.add_trace(go.Box(
                        y=proba_df[class_name],
                        name=class_name,
                        boxpoints='outliers'
                    ))
                fig.update_layout(
                    title="Distribution des probabilités par classe",
                    yaxis_title="Probabilité",
                    xaxis_title="Classe"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Alertes
                if accuracy < 80:
                    st.error(f"⚠️ Taux de détection faible ({accuracy:.2f}%). Le modèle a des difficultés à détecter {true_label}.")
                elif accuracy >= 95:
                    st.success(f"✅ Excellent taux de détection ({accuracy:.2f}%) pour {true_label}.")
                else:
                    st.warning(f"⚠️ Taux de détection modéré ({accuracy:.2f}%) pour {true_label}.")
            
            else:  # test_type == 'benign'
                # Pour le trafic bénin
                false_positives = (y_pred != 'Benign').sum()
                fp_rate = false_positives / len(y_pred) * 100
                
                with col1:
                    st.metric("Faux positifs", f"{false_positives}/{len(y_pred)}")
                with col2:
                    st.metric("Taux de faux positifs", f"{fp_rate:.2f}%")
                with col3:
                    st.metric("Détections correctes", f"{(y_pred == 'Benign').sum()}/{len(y_pred)}")
                with col4:
                    st.metric("Confiance moyenne", f"{y_proba.max(axis=1).mean():.2%}")
                
                # Distribution des prédictions
                st.subheader("📊 Distribution des prédictions")
                pred_counts = pd.Series(y_pred).value_counts()
                fig = px.bar(
                    x=pred_counts.index,
                    y=pred_counts.values,
                    labels={'x': 'Classe prédite', 'y': 'Nombre'},
                    title="Prédictions pour le trafic bénin"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Alertes
                if fp_rate > 10:
                    st.error(f"⚠️ Taux de faux positifs élevé ({fp_rate:.2f}%). Le modèle classe incorrectement le trafic bénin comme attaque.")
                elif fp_rate < 1:
                    st.success(f"✅ Excellent! Très faible taux de faux positifs ({fp_rate:.2f}%).")
                else:
                    st.warning(f"⚠️ Taux de faux positifs modéré ({fp_rate:.2f}%).")
                
                # Détails des faux positifs
                if false_positives > 0:
                    st.subheader("🔍 Détails des faux positifs")
                    fp_indices = np.where(y_pred != 'Benign')[0]
                    fp_data = []
                    for idx in fp_indices[:20]:  # Limiter à 20
                        fp_data.append({
                            'Index': idx,
                            'Prédit': y_pred[idx],
                            'Probabilité': f"{y_proba[idx].max():.2%}",
                            'Top 3 classes': ', '.join([
                                f"{label_encoder.classes_[i]}({y_proba[idx][i]:.2%})"
                                for i in np.argsort(y_proba[idx])[-3:][::-1]
                            ])
                        })
                    st.dataframe(pd.DataFrame(fp_data), use_container_width=True)

if __name__ == "__main__":
    main()

