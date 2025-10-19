import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Reconocimiento de Números Escritos",
    page_icon="🔢",
    layout="wide"
)

class NumberRecognizer:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    @st.cache_resource
    def load_or_train_model(_self):
        """Carga un modelo pre-entrenado o entrena uno nuevo"""
        try:
            # Intentar cargar modelo pre-entrenado
            _self.model = keras.models.load_model('mnist_model.h5')
            _self.is_trained = True
            return _self.model, True
        except:
            with st.spinner('Entrenando modelo... Esto puede tomar unos minutos...'):
                return _self.train_model()
    
    def train_model(self):
        """Entrena el modelo con el dataset MNIST"""
        # Cargar dataset MNIST (números escritos a mano)
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Preprocesar los datos
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Redimensionar para CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Crear modelo CNN
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(2,2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compilar modelo
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar modelo con barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(5):
            status_text.text(f'Entrenando época {epoch+1}/5...')
            history = self.model.fit(
                x_train, y_train,
                epochs=1,
                validation_data=(x_test, y_test),
                batch_size=128,
                verbose=0
            )
            progress_bar.progress((epoch + 1) / 5)
        
        # Guardar modelo
        self.model.save('mnist_model.h5')
        self.is_trained = True
        
        # Mostrar precisión
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        status_text.text(f'Modelo entrenado - Precisión: {test_acc:.4f}')
        
        return self.model, True
    
    def preprocess_image(self, image):
        """Preprocesa la imagen para el modelo"""
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Invertir colores si es necesario (fondo blanco, número negro)
        if np.mean(image) > 127:
            image = 255 - image
        
        # Redimensionar a 28x28 (tamaño MNIST)
        image = cv2.resize(image, (28, 28))
        
        # Normalizar
        image = image.astype('float32') / 255.0
        
        # Preparar para el modelo
        image = image.reshape(1, 28, 28, 1)
        
        return image
    
    def predict_number(self, image):
        """Predice el número en la imagen"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        # Preprocesar imagen
        processed_image = self.preprocess_image(image)
        
        # Predecir
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_number, confidence, prediction[0]
    
    def predict_multiple_numbers(self, image):
        """Detecta y reconoce múltiples números en una imagen"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        # Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Preprocesar para detección de contornos
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        numbers = []
        result_image = image.copy()
        
        for contour in contours:
            # Obtener rectángulo del contorno
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar contornos muy pequeños
            if w > 20 and h > 20:
                # Extraer ROI (Region of Interest)
                roi = gray[y:y+h, x:x+w]
                
                # Preprocesar ROI
                if np.mean(roi) > 127:
                    roi = 255 - roi
                
                roi = cv2.resize(roi, (28, 28))
                roi = roi.astype('float32') / 255.0
                roi = roi.reshape(1, 28, 28, 1)
                
                # Predecir
                prediction = self.model.predict(roi, verbose=0)
                number = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > 0.5:  # Solo considerar predicciones confiables
                    numbers.append({
                        'number': number,
                        'confidence': confidence,
                        'position': (x, y, w, h),
                        'predictions': prediction[0]
                    })
                    
                    # Dibujar rectángulo y texto en la imagen original
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(result_image, f'{number} ({confidence:.2f})', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return numbers, result_image

def main():
    # Título de la aplicación
    st.title("🔢 Reconocimiento de Números Escritos a Mano")
    st.markdown("---")
    
    # Inicializar reconocedor
    recognizer = NumberRecognizer()
    
    # Cargar o entrenar modelo
    with st.spinner('Cargando modelo de IA...'):
        model, loaded = recognizer.load_or_train_model()
    
    if loaded:
        st.success("✅ Modelo de IA cargado exitosamente!")
    
    # Sidebar para configuración
    st.sidebar.title("Configuración")
    recognition_mode = st.sidebar.radio(
        "Modo de reconocimiento:",
        ["Reconocimiento Simple", "Detección Múltiple"]
    )
    
    confidence_threshold = st.sidebar.slider(
        "Umbral de confianza:",
        min_value=0.5,
        max_value=0.99,
        value=0.7,
        step=0.05
    )
    
    # Área para subir archivo
    st.subheader("📤 Subir Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen con números escritos:",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos soportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen subida
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen Original")
            # Convertir a array de numpy
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            st.subheader("Resultados")
            
            if recognition_mode == "Reconocimiento Simple":
                try:
                    with st.spinner('Reconociendo número...'):
                        number, confidence, all_predictions = recognizer.predict_number(image_array)
                    
                    if confidence >= confidence_threshold:
                        st.success(f"**Número reconocido: {number}**")
                        st.info(f"**Confianza: {confidence:.4f}**")
                        
                        # Mostrar gráfico de probabilidades
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(range(10), all_predictions, color='skyblue')
                        bars[number].set_color('red')
                        ax.set_xlabel('Números')
                        ax.set_ylabel('Probabilidad')
                        ax.set_title('Probabilidades de Predicción')
                        ax.set_xticks(range(10))
                        ax.set_ylim(0, 1)
                        
                        # Añadir valores en las barras
                        for i, v in enumerate(all_predictions):
                            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                    else:
                        st.warning(f"Confianza baja ({confidence:.4f}). Intenta con una imagen más clara.")
                        
                except Exception as e:
                    st.error(f"Error en el reconocimiento: {str(e)}")
            
            else:  # Detección Múltiple
                try:
                    with st.spinner('Buscando y reconociendo números...'):
                        numbers, result_image = recognizer.predict_multiple_numbers(image_array)
                    
                    if numbers:
                        st.success(f"✅ Se encontraron {len(numbers)} número(s)")
                        
                        # Mostrar imagen con detecciones
                        st.image(result_image, caption="Detecciones", use_column_width=True)
                        
                        # Mostrar resultados en tabla
                        st.subheader("📊 Resultados Detallados")
                        results_data = []
                        for i, num_info in enumerate(numbers, 1):
                            if num_info['confidence'] >= confidence_threshold:
                                results_data.append({
                                    'Número': num_info['number'],
                                    'Confianza': f"{num_info['confidence']:.4f}",
                                    'Posición': f"({num_info['position'][0]}, {num_info['position'][1]})"
                                })
                        
                        if results_data:
                            st.table(results_data)
                        else:
                            st.warning("No se encontraron números con confianza suficiente.")
                            
                    else:
                        st.warning("No se detectaron números en la imagen.")
                        
                except Exception as e:
                    st.error(f"Error en la detección múltiple: {str(e)}")
    
    # Información adicional
    with st.expander("💡 Consejos para mejores resultados"):
        st.markdown("""
        - **Imágenes claras**: Usa imágenes con buen contraste
        - **Fondo simple**: Fondo blanco o claro funciona mejor
        - **Números centrados**: Un número por imagen para reconocimiento simple
        - **Tamaño adecuado**: Los números deben ser visibles y claros
        - **Formato**: JPG o PNG recomendados
        """)
    
    with st.expander("📊 Información del Modelo"):
        st.markdown("""
        - **Arquitectura**: Red Neuronal Convolucional (CNN)
        - **Dataset**: MNIST (70,000 imágenes de números escritos a mano)
        - **Precisión**: >98% en el dataset de prueba
        - **Entrada**: Imágenes 28x28 píxeles en escala de grises
        """)

if __name__ == "__main__":
    main()
