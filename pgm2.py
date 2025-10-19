import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import io
import os

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Reconocimiento de Números Escritos",
    page_icon="🔢",
    layout="wide"
)

# Ocultar advertencias de TensorFlow
tf.get_logger().setLevel('ERROR')

class NumberRecognizer:
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    @st.cache_resource
    def load_model(_self):
        """Carga o crea el modelo directamente sin guardar archivos"""
        try:
            # Intentar cargar el dataset y crear modelo
            with st.spinner('📥 Cargando dataset MNIST...'):
                (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            # Preprocesar datos
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)
            
            # Crear modelo
            _self.model = keras.Sequential([
                keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Conv2D(64, (3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            # Compilar
            _self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entrenar rápidamente
            with st.spinner('🚀 Entrenando modelo rápido...'):
                _self.model.fit(
                    x_train, y_train,
                    epochs=2,
                    batch_size=512,
                    validation_data=(x_test, y_test),
                    verbose=0
                )
            
            _self.is_trained = True
            st.success("✅ Modelo listo!")
            return _self.model, True
            
        except Exception as e:
            st.error(f"❌ Error cargando modelo: {e}")
            # Crear modelo simple de emergencia
            return _self.create_emergency_model()
    
    def create_emergency_model(_self):
        """Crea un modelo muy simple como último recurso"""
        try:
            # Modelo minimalista
            _self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            _self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Datos dummy para entrenamiento rápido
            dummy_data = np.random.random((100, 28, 28))
            dummy_labels = np.random.randint(0, 10, 100)
            
            _self.model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)
            _self.is_trained = True
            
            st.warning("⚠️ Usando modelo de emergencia - precisión limitada")
            return _self.model, True
            
        except Exception as e:
            st.error(f"❌ Error crítico: {e}")
            return None, False

    def preprocess_image(self, image):
        """Preprocesa la imagen para el modelo"""
        try:
            # Convertir a escala de grises
            if image.mode != 'L':
                image = image.convert('L')
            
            # Redimensionar a 28x28
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            image_array = np.array(image).astype('float32') / 255.0
            
            # Invertir si el fondo es blanco
            if np.mean(image_array) > 0.5:
                image_array = 1.0 - image_array
            
            # Asegurar rango correcto
            image_array = np.clip(image_array, 0, 1)
            
            return image_array.reshape(1, 28, 28, 1)
            
        except Exception as e:
            st.error(f"Error preprocesando imagen: {e}")
            raise

    def predict_number(self, image):
        """Predice el número en la imagen"""
        if self.model is None:
            raise ValueError("Modelo no disponible")
        
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_number, confidence, prediction[0]

    def process_entire_image(self, image):
        """Procesa la imagen completa buscando números"""
        if self.model is None:
            raise ValueError("Modelo no disponible")
        
        # Convertir a escala de grises
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        
        results = []
        img_width, img_height = gray_image.size
        
        # Estrategia 1: Procesar imagen completa si es pequeña
        if img_width <= 150 and img_height <= 150:
            try:
                number, confidence, _ = self.predict_number(gray_image)
                if confidence > 0.1:
                    results.append({
                        'number': number,
                        'confidence': confidence,
                        'position': (0, 0, img_width, img_height),
                        'region': gray_image
                    })
            except:
                pass
        
        # Estrategia 2: Dividir en cuadrícula
        if not results:
            grid_size = 4
            cell_width = img_width // grid_size
            cell_height = img_height // grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * cell_width
                    y1 = i * cell_height
                    x2 = min((j + 1) * cell_width, img_width)
                    y2 = min((i + 1) * cell_height, img_height)
                    
                    if (x2 - x1) > 20 and (y2 - y1) > 20:  # Tamaño mínimo
                        cell = gray_image.crop((x1, y1, x2, y2))
                        
                        try:
                            number, confidence, _ = self.predict_number(cell)
                            if confidence > 0.1:
                                results.append({
                                    'number': number,
                                    'confidence': confidence,
                                    'position': (x1, y1, x2-x1, y2-y1),
                                    'region': cell
                                })
                        except:
                            continue
        
        # Ordenar por confianza y eliminar duplicados
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filtrar duplicados
        final_results = []
        for result in results:
            is_duplicate = False
            for final in final_results:
                if (abs(result['number'] - final['number']) == 0 and
                    abs(result['position'][0] - final['position'][0]) < 50 and
                    abs(result['position'][1] - final['position'][1]) < 50):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_results.append(result)
        
        return final_results[:6]  # Máximo 6 resultados

def main():
    st.title("🔢 Reconocimiento de Números Escritos a Mano")
    st.markdown("---")
    
    # Inicializar en la sesión
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = NumberRecognizer()
    
    if 'model_loaded' not in st.session_state:
        with st.spinner('🚀 Inicializando IA...'):
            model, loaded = st.session_state.recognizer.load_model()
            st.session_state.model_loaded = loaded
    
    if not st.session_state.model_loaded:
        st.error("❌ No se pudo inicializar el modelo. Recarga la página.")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    mode = st.sidebar.radio(
        "**Modo:**",
        ["Reconocimiento Simple", "Búsqueda Múltiple"]
    )
    
    confidence = st.sidebar.slider(
        "**Umbral de confianza:**",
        0.1, 0.99, 0.5, 0.05
    )
    
    # Upload
    st.subheader("📤 Subir Imagen")
    uploaded_file = st.file_uploader(
        "**Arrastra y suelta tu imagen aquí**",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Imagen Original")
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error cargando imagen: {e}")
                return
        
        with col2:
            st.subheader("📊 Resultados")
            
            try:
                if mode == "Reconocimiento Simple":
                    with st.spinner('🔍 Analizando...'):
                        number, conf, probs = st.session_state.recognizer.predict_number(image)
                    
                    if conf >= confidence:
                        st.success(f"**✅ Número: {number}**")
                        st.info(f"**🎯 Confianza: {conf:.1%}**")
                        
                        # Gráfico
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['red' if i == number else 'blue' for i in range(10)]
                        ax.bar(range(10), probs, color=colors, alpha=0.7)
                        ax.set_xlabel('Números (0-9)')
                        ax.set_ylabel('Probabilidad')
                        ax.set_xticks(range(10))
                        
                        for i, v in enumerate(probs):
                            ax.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig)
                    else:
                        st.warning(f"Confianza baja ({conf:.1%}). Prueba:")
                        st.markdown("- Imagen más nítida")
                        st.markdown("- Mejor contraste")
                        st.markdown("- Número centrado")
                
                else:  # Búsqueda múltiple
                    with st.spinner('🔍 Buscando números...'):
                        results = st.session_state.recognizer.process_entire_image(image)
                    
                    valid_results = [r for r in results if r['confidence'] >= confidence]
                    
                    if valid_results:
                        st.success(f"✅ Encontrados: {len(valid_results)} números")
                        
                        # Tabla
                        st.subheader("📋 Resultados")
                        for i, result in enumerate(valid_results, 1):
                            st.write(f"**{i}. Número {result['number']}** - Confianza: {result['confidence']:.1%}")
                        
                        # Regiones
                        st.subheader("🔍 Áreas Detectadas")
                        cols = st.columns(2)
                        for i, result in enumerate(valid_results):
                            with cols[i % 2]:
                                st.image(
                                    result['region'],
                                    caption=f'Número: {result["number"]} ({result["confidence"]:.1%})',
                                    use_container_width=True
                                )
                    else:
                        st.warning("No se encontraron números con confianza suficiente")
                        st.markdown("**Sugerencias:**")
                        st.markdown("- Reduce el umbral de confianza")
                        st.markdown("- Verifica que los números sean visibles")
                        st.markdown("- Prueba el modo Simple")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 **Soluciones:**")
                st.markdown("- Usa modo 'Reconocimiento Simple'")
                st.markdown("- Imagen con mejor calidad")
                st.markdown("- Recarga la página si persiste")
    
    else:
        # Instrucciones
        st.info("👆 **Cómo usar:**")
        st.markdown("""
        1. **Sube una imagen** con números
        2. **Selecciona el modo** de reconocimiento  
        3. **Ajusta la confianza** si es necesario
        4. **¡Obtén resultados!**
        """)
        
        st.markdown("**💡 Para mejores resultados:**")
        st.markdown("- Fondo claro, números oscuros")
        st.markdown("- Imágenes nítidas")
        st.markdown("- Buena iluminación")

if __name__ == "__main__":
    main()
