import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import io

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
        """Preprocesa la imagen para el modelo usando PIL"""
        # Convertir a escala de grises si es necesario
        if image.mode != 'L':
            image = image.convert('L')
        
        # Invertir colores si es necesario (fondo blanco, número negro)
        image_array = np.array(image)
        if np.mean(image_array) > 127:
            image = ImageOps.invert(image)
        
        # Redimensionar a 28x28 (tamaño MNIST)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir a array y normalizar
        image_array = np.array(image).astype('float32') / 255.0
        
        # Preparar para el modelo
        image_array = image_array.reshape(1, 28, 28, 1)
        
        return image_array
    
    def preprocess_image_advanced(self, image):
        """Preprocesamiento más avanzado para mejorar el reconocimiento"""
        # Convertir a escala de grises
        if image.mode != 'L':
            image = image.convert('L')
        
        # Aplicar filtro para suavizar
        image = image.filter(ImageFilter.SMOOTH)
        
        # Convertir a array numpy
        img_array = np.array(image)
        
        # Invertir si el fondo es blanco
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalizar y ajustar contraste
        img_array = img_array.astype('float32')
        if img_array.max() > 0:
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        
        # Redimensionar
        img_pil = Image.fromarray((img_array * 255).astype('uint8'))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convertir de nuevo a array
        final_array = np.array(img_pil).astype('float32') / 255.0
        final_array = final_array.reshape(1, 28, 28, 1)
        
        return final_array
    
    def predict_number(self, image):
        """Predice el número en la imagen"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        # Preprocesar imagen
        processed_image = self.preprocess_image_advanced(image)
        
        # Predecir
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_number, confidence, prediction[0]
    
    def find_numbers_in_image(self, image):
        """Encuentra números en diferentes regiones de la imagen"""
        if not self.is_trained:
            raise ValueError("El modelo no está entrenado")
        
        # Convertir a escala de grises
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image.copy()
        
        # Crear una copia para dibujar resultados
        result_image = image.convert('RGB')
        
        # Dividir la imagen en regiones para buscar números
        img_width, img_height = gray_image.size
        numbers_found = []
        
        # Probar diferentes tamaños de regiones
        region_sizes = [min(img_width, img_height) // 2, min(img_width, img_height) // 3]
        
        for region_size in region_sizes:
            for y in range(0, img_height, region_size // 2):
                for x in range(0, img_width, region_size // 2):
                    # Extraer región
                    region = gray_image.crop((
                        max(0, x),
                        max(0, y),
                        min(img_width, x + region_size),
                        min(img_height, y + region_size)
                    ))
                    
                    # Verificar si la región tiene contenido
                    region_array = np.array(region)
                    if np.std(region_array) > 20:  # Si hay variación (posible número)
                        try:
                            number, confidence, _ = self.predict_number(region)
                            if confidence > 0.5:
                                numbers_found.append({
                                    'number': number,
                                    'confidence': confidence,
                                    'position': (x, y, region_size, region_size),
                                    'region': region
                                })
                        except:
                            continue
        
        # Eliminar duplicados
        unique_numbers = []
        for num in numbers_found:
            is_duplicate = False
            for unique in unique_numbers:
                dist = np.sqrt((num['position'][0] - unique['position'][0])**2 + 
                              (num['position'][1] - unique['position'][1])**2)
                if dist < 50 and num['number'] == unique['number']:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_numbers.append(num)
        
        return unique_numbers

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
        ["Reconocimiento Simple", "Búsqueda en Regiones"]
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
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen subida", use_column_width=True)
        
        with col2:
            st.subheader("Resultados")
            
            if recognition_mode == "Reconocimiento Simple":
                try:
                    with st.spinner('Reconociendo número...'):
                        number, confidence, all_predictions = recognizer.predict_number(image)
                    
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
            
            else:  # Búsqueda en Regiones
                try:
                    with st.spinner('Buscando números en la imagen...'):
                        numbers_found = recognizer.find_numbers_in_image(image)
                    
                    if numbers_found:
                        st.success(f"✅ Se encontraron {len(numbers_found)} número(s)")
                        
                        # Mostrar resultados en tabla
                        st.subheader("📊 Resultados Detallados")
                        results_data = []
                        for i, num_info in enumerate(numbers_found, 1):
                            if num_info['confidence'] >= confidence_threshold:
                                results_data.append({
                                    'Número': num_info['number'],
                                    'Confianza': f"{num_info['confidence']:.4f}",
                                    'Posición': f"({num_info['position'][0]}, {num_info['position'][1]})"
                                })
                        
                        if results_data:
                            st.table(results_data)
                            
                            # Mostrar regiones encontradas
                            st.subheader("🔍 Regiones Encontradas")
                            cols = st.columns(3)
                            for i, num_info in enumerate(numbers_found):
                                if num_info['confidence'] >= confidence_threshold:
                                    with cols[i % 3]:
                                        st.image(
                                            num_info['region'], 
                                            caption=f'Número: {num_info["number"]} (Conf: {num_info["confidence"]:.2f})',
                                            use_column_width=True
                                        )
                        else:
                            st.warning("No se encontraron números con confianza suficiente.")
                            
                    else:
                        st.warning("No se detectaron números en la imagen.")
                        
                except Exception as e:
                    st.error(f"Error en la búsqueda: {str(e)}")
    
    # Información adicional
    with st.expander("💡 Consejos para mejores resultados"):
        st.markdown("""
        - **Imágenes claras**: Usa imágenes con buen contraste
        - **Fondo simple**: Fondo blanco o claro funciona mejor
        - **Números centrados**: Un número por imagen para reconocimiento simple
        - **Tamaño adecuado**: Los números deben ser visibles y claros
        - **Formato**: JPG o PNG recomendados
        - **Para mejores resultados**: Recorta la imagen para que cada número esté centrado
        """)
    
    with st.expander("📊 Información del Modelo"):
        st.markdown("""
        - **Arquitectura**: Red Neuronal Convolucional (CNN)
        - **Dataset**: MNIST (70,000 imágenes de números escritos a mano)
        - **Precisión**: >98% en el dataset de prueba
        - **Entrada**: Imágenes 28x28 píxeles en escala de grises
        - **Tecnología**: TensorFlow/Keras
        """)

if __name__ == "__main__":
    main()
