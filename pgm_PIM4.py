import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import tempfile
import os
import re
import hashlib
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página con medidas de seguridad
st.set_page_config(
    page_title="Sistema de Polaridad Evolutivo Avanzado",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔐 CONFIGURACIÓN DE SEGURIDAD
class SecurityConfig:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB máximo
    ALLOWED_EXTENSIONS = {'.fasta', '.fa', '.txt'}
    MAX_SEQUENCES = 1000
    MAX_SEQUENCE_LENGTH = 10000
    
    @staticmethod
    def validate_file_upload(uploaded_file):
        """Valida archivos subidos"""
        if uploaded_file.size > SecurityConfig.MAX_FILE_SIZE:
            raise ValueError(f"Archivo demasiado grande. Máximo: {SecurityConfig.MAX_FILE_SIZE//1024//1024}MB")
        
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in SecurityConfig.ALLOWED_EXTENSIONS:
            raise ValueError(f"Extensión no permitida. Use: {', '.join(SecurityConfig.ALLOWED_EXTENSIONS)}")
        
        return True

# Grupos de polaridad (idéntico al original)
POLARITY_GROUPS = {
    'P+': ['R', 'H', 'K'],
    'P-': ['D', 'E'],  
    'N': ['S', 'T', 'N', 'Q', 'C', 'Y', 'W'],
    'NP': ['A', 'V', 'I', 'L', 'M', 'F', 'P', 'G']
}

AA_TO_GROUP = {}
for group, amino_acids in POLARITY_GROUPS.items():
    for aa in amino_acids:
        AA_TO_GROUP[aa] = group

GROUP_ORDER = ['P+', 'P-', 'N', 'NP']
GROUP_TO_NUM = {group: i+1 for i, group in enumerate(GROUP_ORDER)}

# 🧠 MEJORA 4: SISTEMA DE AUTO-EVOLUCIÓN
class EvolucionadorReglas:
    def __init__(self):
        self.historico_reglas = []
        self.efectividad_reglas = {}
        self.ultima_actualizacion = None
        
    def evaluar_efectividad_regla(self, regla, aciertos, total):
        """Evalúa la efectividad de una regla específica"""
        if total == 0:
            return 0.0
        
        efectividad = aciertos / total
        clave_regla = f"{regla['posicion']}_{regla['valor']}"
        self.efectividad_reglas[clave_regla] = efectividad
        return efectividad
    
    def mutar_regla_inefectiva(self, regla):
        """Aplica mutación controlada a reglas poco efectivas"""
        regla_mutada = regla.copy()
        
        # Mutación suave: ajustar posición o valor
        if np.random.random() < 0.3:
            regla_mutada['posicion'] = max(1, min(16, regla_mutada['posicion'] + np.random.choice([-1, 1])))
        if np.random.random() < 0.3:
            regla_mutada['valor'] = max(1, min(16, regla_mutada['valor'] + np.random.choice([-1, 1])))
        
        return regla_mutada
    
    def evolucionar_reglas(self, reglas_actuales, tasa_mutacion=0.1):
        """Evoluciona el conjunto de reglas"""
        nuevas_reglas = []
        
        for regla in reglas_actuales:
            clave = f"{regla['posicion']}_{regla['valor']}"
            efectividad = self.efectividad_reglas.get(clave, 0.5)
            
            if efectividad < 0.3 and np.random.random() < tasa_mutacion:
                # Mutar regla inefectiva
                regla_mutada = self.mutar_regla_inefectiva(regla)
                nuevas_reglas.append(regla_mutada)
            elif efectividad > 0.7:
                # Mantener regla efectiva
                nuevas_reglas.append(regla)
            else:
                # Decisión aleatoria para reglas medianas
                if np.random.random() < 0.7:
                    nuevas_reglas.append(regla)
        
        self.ultima_actualizacion = datetime.now()
        return nuevas_reglas

# 🧠 MEJORA 5: DETECCIÓN DE PATRONES COMPUESTOS
class DetectorPatronesCompuestos:
    def __init__(self):
        self.patrones_compuestos = []
        self.ventana_analisis = 3
        
    def extraer_patrones_compuestos(self, secuencias):
        """Extrae patrones que involucran múltiples posiciones"""
        patrones_encontrados = []
        
        for sec_id, secuencia in list(secuencias.items())[:50]:  # Muestra para eficiencia
            lineal = self._secuencia_a_lineal(secuencia)
            
            # Buscar patrones en ventana deslizante
            for i in range(len(lineal) - self.ventana_analisis + 1):
                ventana = tuple(lineal[i:i + self.ventana_analisis])
                if all(v > 0 for v in ventana):  # Solo posiciones válidas
                    patrones_encontrados.append(ventana)
        
        # Contar frecuencia de patrones
        from collections import Counter
        contador = Counter(patrones_encontrados)
        
        # Filtrar patrones frecuentes
        self.patrones_compuestos = [
            patron for patron, count in contador.items() 
            if count >= 2  # Al menos 2 ocurrencias
        ]
        
        return len(self.patrones_compuestos)
    
    def evaluar_patron_compuesto(self, secuencia):
        """Evalúa si una secuencia contiene patrones compuestos conocidos"""
        lineal = self._secuencia_a_lineal(secuencia)
        coincidencias = 0
        
        for patron in self.patrones_compuestos[:10]:  # Usar solo los 10 más frecuentes
            for i in range(len(lineal) - len(patron) + 1):
                if tuple(lineal[i:i + len(patron)]) == patron:
                    coincidencias += 1
                    break
        
        return min(1.0, coincidencias / 10)  # Normalizar a 0-1
    
    def _secuencia_a_lineal(self, sequence):
        """Convierte secuencia a formato lineal"""
        lineal = np.zeros(16, dtype=int)
        valid_sequence = [aa for aa in sequence if aa in AA_TO_GROUP]
        
        for i in range(min(16, len(valid_sequence))):
            aa = valid_sequence[i]
            grupo = AA_TO_GROUP[aa]
            lineal[i] = GROUP_TO_NUM[grupo]
        
        return lineal

# 🧠 MEJORA 6: SISTEMA DE ALERTAS TEMPRANAS
class SistemaAlertas:
    def __init__(self):
        self.historico_confianzas = []
        self.alertas_activas = []
        self.umbral_deriva = 0.15
        
    def monitorizar_desempeno(self, confianza_actual):
        """Monitorea el desempeño en tiempo real"""
        self.historico_confianzas.append(confianza_actual)
        
        if len(self.historico_confianzas) < 10:
            return []  # Necesitamos suficiente historial
        
        # Calcular tendencia
        ventana_reciente = self.historico_confianzas[-5:]
        ventana_antigua = self.historico_confianzas[-10:-5]
        
        if len(ventana_antigua) > 0:
            prom_reciente = np.mean(ventana_reciente)
            prom_antigua = np.mean(ventana_antigua)
            
            # Detectar deriva de concepto
            if prom_antigua - prom_reciente > self.umbral_deriva * 100:
                alerta = {
                    'tipo': 'deriva_concepto',
                    'severidad': 'alta',
                    'mensaje': 'Posible deriva de concepto detectada',
                    'timestamp': datetime.now()
                }
                if alerta not in self.alertas_activas:
                    self.alertas_activas.append(alerta)
        
        return self.alertas_activas
    
    def limpiar_alertas_antiguas(self):
        """Limpia alertas antiguas"""
        ahora = datetime.now()
        self.alertas_activas = [
            alerta for alerta in self.alertas_activas
            if (ahora - alerta['timestamp']).total_seconds() < 3600  # 1 hora
        ]

# CLASE BASE MEJORADA
class PolaridadReplica:
    def __init__(self):
        self.perfil_maestro = None
        self.perfiles_otros = {}
        self.reglas_discriminantes = []
        self.viene3_code = None
        self.entrenado = False
        self.matriz_maestro = None
        self.matriz_contraste = None
        
    def parse_fasta(self, fasta_text):
        """Parsea FASTA con validación de seguridad"""
        sequences = {}
        current_id = None
        current_seq = []
        sequence_count = 0
        
        for line in fasta_text.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                    sequence_count += 1
                    if sequence_count >= SecurityConfig.MAX_SEQUENCES:
                        st.warning(f"Límite de {SecurityConfig.MAX_SEQUENCES} secuencias alcanzado")
                        break
                current_id = line[1:].split()[0]
                current_seq = []
            elif line:
                # Validar caracteres de aminoácidos
                line_clean = ''.join(c for c in line.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
                if len(line_clean) > SecurityConfig.MAX_SEQUENCE_LENGTH:
                    st.warning(f"Secuencia truncada a {SecurityConfig.MAX_SEQUENCE_LENGTH} aminoácidos")
                    line_clean = line_clean[:SecurityConfig.MAX_SEQUENCE_LENGTH]
                current_seq.append(line_clean)
        
        if current_id is not None and sequence_count < SecurityConfig.MAX_SEQUENCES:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def calcular_matriz_polaridad(self, sequence):
        """Calcula matriz 4x4 idéntica al Fortran original"""
        matrix = np.zeros((4, 4), dtype=int)
        valid_sequence = [aa for aa in sequence if aa in AA_TO_GROUP]
        
        for i in range(len(valid_sequence) - 1):
            current_aa = valid_sequence[i]
            next_aa = valid_sequence[i + 1]
            
            current_group = AA_TO_GROUP[current_aa]
            next_group = AA_TO_GROUP[next_aa]
            
            row_idx = GROUP_ORDER.index(current_group)
            col_idx = GROUP_ORDER.index(next_group)
            
            matrix[row_idx, col_idx] += 1
        
        return matrix
    
    def calcular_matriz_incidencias_grupo(self, secuencias):
        """Calcula la matriz de incidencias para un grupo de secuencias"""
        matriz_acumulada = np.zeros((4, 4), dtype=int)
        
        for secuencia in secuencias.values():
            matriz_secuencia = self.calcular_matriz_polaridad(secuencia)
            matriz_acumulada += matriz_secuencia
        
        return matriz_acumulada
    
    def secuencia_a_lineal(self, sequence, longitud=16):
        """Convierte secuencia a formato lineal (16 posiciones) como el Fortran"""
        lineal = np.zeros(longitud, dtype=int)
        valid_sequence = [aa for aa in sequence if aa in AA_TO_GROUP]
        
        for i in range(min(longitud, len(valid_sequence))):
            aa = valid_sequence[i]
            grupo = AA_TO_GROUP[aa]
            lineal[i] = GROUP_TO_NUM[grupo]
        
        return lineal
    
    def entrenar_sistema_original(self, secuencias_maestro, secuencias_contraste):
        """Entrenamiento completo del sistema original"""
        # Calcular matrices de incidencias
        self.matriz_maestro = self.calcular_matriz_incidencias_grupo(secuencias_maestro)
        self.matriz_contraste = self.calcular_matriz_incidencias_grupo(secuencias_contraste)
        
        # Simular archivos temporales
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.net') as f:
            for seq_id, secuencia in list(secuencias_maestro.items()):
                linealA = self.secuencia_a_lineal(secuencia)
                linea = f"PRT " + " ".join(f"{x:2d}" for x in linealA) + "  1 100"
                f.write(linea + '\n')
            archivo_maestro = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.net') as f:
            for seq_id, secuencia in list(secuencias_contraste.items()):
                linealA = self.secuencia_a_lineal(secuencia)
                linea = f"PRT " + " ".join(f"{x:2d}" for x in linealA) + "  0 100"
                f.write(linea + '\n')
            archivo_contraste = f.name
        
        # Coincidencias
        maestro, reglas_maestro = self.coincidencias_maestro(archivo_maestro)
        otros, reglas_otros = self.coincidencias_otros(archivo_contraste)
        
        # Discriminante
        diferencias = self.discriminante(maestro, otros)
        self.reglas_discriminantes = diferencias
        
        # Limpiar
        try:
            os.unlink(archivo_maestro)
            os.unlink(archivo_contraste)
        except:
            pass
        
        return len(reglas_maestro), len(reglas_otros), len(diferencias)
    
    def coincidencias_maestro(self, archivo_objetivo):
        """Réplica exacta de coincidenciasmaestro.f"""
        maestro = np.zeros((16, 16), dtype=int)
        
        try:
            with open(archivo_objetivo, 'r') as f:
                for linea in f:
                    if len(linea.strip()) < 10:
                        continue
                    
                    partes = linea.strip().split()
                    if len(partes) < 18:
                        continue
                    
                    try:
                        linealA = [int(x) for x in partes[1:17]]
                        for i, valor in enumerate(linealA):
                            if 1 <= valor <= 16:
                                maestro[valor-1, i] += 1
                    except (ValueError, IndexError):
                        continue
        except FileNotFoundError:
            return maestro, []
        
        reglas = []
        for i in range(16):
            for valor in range(1, 17):
                if maestro[valor-1, i] == 0:
                    reglas.append(f"if (linealA({i+1}) .eq. {valor}) espe = 0")
        
        return maestro, reglas
    
    def coincidencias_otros(self, archivo_objetivo):
        """Réplica exacta de coincidenciasotros.f"""
        otros = np.zeros((16, 16), dtype=int)
        
        try:
            with open(archivo_objetivo, 'r') as f:
                for linea in f:
                    if len(linea.strip()) < 10:
                        continue
                    
                    partes = linea.strip().split()
                    if len(partes) < 18:
                        continue
                    
                    try:
                        linealA = [int(x) for x in partes[1:17]]
                        for i, valor in enumerate(linealA):
                            if 1 <= valor <= 16:
                                otros[valor-1, i] += 1
                    except (ValueError, IndexError):
                        continue
        except FileNotFoundError:
            return otros, []
        
        reglas = []
        for i in range(16):
            for valor in range(1, 17):
                if otros[valor-1, i] != 0:
                    reglas.append(f"if (linealA({i+1}) .eq. {valor}) espe = 0")
        
        return otros, reglas
    
    def discriminante(self, maestro, otros):
        """Réplica exacta de discriminante.f"""
        total_maestro = np.sum(maestro)
        total_otros = np.sum(otros)
        
        diferencias = []
        
        for i in range(16):
            for j in range(16):
                freq_maestro = maestro[j, i] / total_maestro if total_maestro > 0 else 0
                freq_otros = otros[j, i] / total_otros if total_otros > 0 else 0
                
                if freq_maestro + 0.03 < freq_otros:
                    diferencias.append({
                        'freq_maestro': freq_maestro,
                        'freq_otros': freq_otros,
                        'posicion': i + 1,
                        'valor': j + 1
                    })
        
        diferencias.sort(key=lambda x: x['freq_otros'] - x['freq_maestro'], reverse=True)
        return diferencias
    
    def evaluar_proteina_original(self, secuencia):
        """Evaluación original mejorada"""
        if not self.entrenado:
            return "No entrenado", 0
        
        linealA = self.secuencia_a_lineal(secuencia)
        matriz = self.calcular_matriz_polaridad(secuencia)
        total_transiciones = np.sum(matriz)
        
        if total_transiciones == 0:
            return "No", 0
        
        # Score mejorado
        score = 0.0
        transiciones_validas = 0
        
        valid_sequence = [aa for aa in secuencia if aa in AA_TO_GROUP]
        
        for i in range(len(valid_sequence) - 1):
            current_aa = valid_sequence[i]
            next_aa = valid_sequence[i + 1]
            
            current_group = AA_TO_GROUP[current_aa]
            next_group = AA_TO_GROUP[next_aa]
            
            row_idx = GROUP_ORDER.index(current_group)
            col_idx = GROUP_ORDER.index(next_group)
            
            if row_idx == col_idx:
                score += 0.8
            elif abs(row_idx - col_idx) == 1:
                score += 0.6
            else:
                score += 0.3
                
            transiciones_validas += 1
        
        if transiciones_validas > 0:
            score = score / transiciones_validas
        else:
            score = 0
        
        # Aplicar reglas
        espe = 1
        for diff in self.reglas_discriminantes[:5]:
            pos = diff['posicion'] - 1
            if pos < len(linealA) and linealA[pos] == diff['valor']:
                espe = 0
                break
        
        if espe == 1 and score > 0.4:
            return "Yes", int(score * 100)
        else:
            return "No", int(score * 100)

# 🧠 MEJORA 1: ENSEMBLE DE PERFILES
class EnsemblePolaridad:
    def __init__(self):
        self.perfiles = {}
        self.pesos_confianza = {}
        
    def agregar_perfil(self, nombre, perfil, peso=1.0):
        self.perfiles[nombre] = perfil
        self.pesos_confianza[nombre] = peso
    
    def predecir_ensemble(self, secuencia):
        predicciones = []
        confianzas = []
        
        for nombre, perfil in self.perfiles.items():
            resultado, confianza = perfil.evaluar_proteina_original(secuencia)
            prediccion = 1 if resultado == "Yes" else 0
            predicciones.append(prediccion)
            confianzas.append(confianza * self.pesos_confianza[nombre])
        
        # Voto ponderado por confianza
        if sum(confianzas) > 0:
            voto_ponderado = sum(p * c for p, c in zip(predicciones, confianzas)) / sum(confianzas)
            resultado_final = "Yes" if voto_ponderado > 0.5 else "No"
            confianza_final = int(voto_ponderado * 100)
        else:
            resultado_final = "No"
            confianza_final = 0
            
        return resultado_final, confianza_final

# 🧠 MEJORA 2: SISTEMA DE ATENCIÓN
class SistemaAtencion:
    def __init__(self):
        self.posiciones_importantes = {}
        
    def analizar_importancia_posiciones(self, secuencias_maestro, secuencias_contraste):
        """Identifica posiciones más importantes usando información mutua"""
        importancia = np.zeros(16)
        
        for secuencias, peso in [(secuencias_maestro, 1.0), (secuencias_contraste, -0.5)]:
            for secuencia in secuencias.values():
                lineal = self._secuencia_a_lineal(secuencia)
                for i, valor in enumerate(lineal):
                    if valor > 0:
                        importancia[i] += peso
        
        # Normalizar y guardar posiciones importantes
        if np.max(np.abs(importancia)) > 0:
            importancia_normalizada = importancia / np.max(np.abs(importancia))
            self.posiciones_importantes = {
                i: imp for i, imp in enumerate(importancia_normalizada) 
                if abs(imp) > 0.3
            }
        
        return len(self.posiciones_importantes)
    
    def _secuencia_a_lineal(self, sequence):
        """Convierte secuencia a formato lineal"""
        lineal = np.zeros(16, dtype=int)
        valid_sequence = [aa for aa in sequence if aa in AA_TO_GROUP]
        
        for i in range(min(16, len(valid_sequence))):
            aa = valid_sequence[i]
            grupo = AA_TO_GROUP[aa]
            lineal[i] = GROUP_TO_NUM[grupo]
        
        return lineal
    
    def aplicar_atencion(self, secuencia, resultado_base, confianza_base):
        """Aplica pesos de atención al resultado base"""
        if not self.posiciones_importantes:
            return resultado_base, confianza_base
        
        lineal = self._secuencia_a_lineal(secuencia)
        factor_ajuste = 1.0
        
        for pos, importancia in self.posiciones_importantes.items():
            if pos < len(lineal) and lineal[pos] > 0:
                # Ajustar confianza basado en posiciones importantes
                factor_ajuste += importancia * 0.2
        
        confianza_ajustada = min(100, max(0, int(confianza_base * factor_ajuste)))
        
        # Si hay posiciones muy importantes que coinciden, reforzar resultado
        if factor_ajuste > 1.1 and resultado_base == "Yes":
            confianza_ajustada = min(100, confianza_ajustada + 10)
        elif factor_ajuste < 0.9 and resultado_base == "Yes":
            confianza_ajustada = max(0, confianza_ajustada - 10)
            
        return resultado_base, confianza_ajustada

# 🧠 SISTEMA EVOLUTIVO COMPLETO
class PolaridadEvolutiva:
    def __init__(self):
        self.sistema_original = PolaridadReplica()
        self.ensemble = EnsemblePolaridad()
        self.sistema_atencion = SistemaAtencion()
        self.evolucionador = EvolucionadorReglas()
        self.detector_patrones = DetectorPatronesCompuestos()
        self.sistema_alertas = SistemaAlertas()
        self.entrenado = False
        self.metricas_desempeno = {
            'aciertos': 0,
            'total_predicciones': 0,
            'confianza_promedio': 0,
            'mejora_vs_original': 0
        }
        self.modelo_hash = None
    
    def calcular_hash_modelo(self, secuencias_maestro, secuencias_contraste):
        """Calcula hash único para el modelo entrenado"""
        contenido = (
            str(sorted(secuencias_maestro.items())) + 
            str(sorted(secuencias_contraste.items())) +
            str(datetime.now())
        )
        self.modelo_hash = hashlib.md5(contenido.encode()).hexdigest()[:12]
        return self.modelo_hash
    
    def entrenar_sistema_completo(self, secuencias_maestro, secuencias_contraste, grupo_maestro):
        """Entrenamiento del sistema híbrido completo"""
        
        # 🔐 Calcular hash de seguridad
        modelo_hash = self.calcular_hash_modelo(secuencias_maestro, secuencias_contraste)
        
        # 1. Entrenar sistema original
        with st.spinner("Fase 1: Entrenando sistema base..."):
            reglas_m, reglas_o, discriminantes = self.sistema_original.entrenar_sistema_original(
                secuencias_maestro, secuencias_contraste
            )
            self.sistema_original.entrenado = True
        
        # 2. Configurar ensemble
        with st.spinner("Fase 2: Configurando ensemble..."):
            self.ensemble.agregar_perfil("base", self.sistema_original, peso=1.0)
        
        # 3. Entrenar sistema de atención
        with st.spinner("Fase 3: Analizando patrones de atención..."):
            pos_importantes = self.sistema_atencion.analizar_importancia_posiciones(
                secuencias_maestro, secuencias_contraste
            )
        
        # 🧠 MEJORA 4: Detectar patrones compuestos
        with st.spinner("Fase 4: Detectando patrones compuestos..."):
            patrones_compuestos = self.detector_patrones.extraer_patrones_compuestos(secuencias_maestro)
        
        self.entrenado = True
        
        return {
            'reglas_maestro': reglas_m,
            'reglas_otros': reglas_o,
            'discriminantes': discriminantes,
            'posiciones_importantes': pos_importantes,
            'patrones_compuestos': patrones_compuestos,
            'modelo_hash': modelo_hash,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def clasificar_evolutivo(self, secuencia):
        """Clasificación usando el sistema híbrido completo"""
        if not self.entrenado:
            return "No entrenado", 0
        
        # 1. Predicción del ensemble
        resultado_ensemble, confianza_ensemble = self.ensemble.predecir_ensemble(secuencia)
        
        # 2. Aplicar atención
        resultado_atencion, confianza_atencion = self.sistema_atencion.aplicar_atencion(
            secuencia, resultado_ensemble, confianza_ensemble
        )
        
        # 🧠 MEJORA 5: Evaluar patrones compuestos
        score_patrones = self.detector_patrones.evaluar_patron_compuesto(secuencia)
        confianza_final = min(100, confianza_atencion + int(score_patrones * 20))
        
        # 🧠 MEJORA 6: Monitorear alertas
        alertas = self.sistema_alertas.monitorizar_desempeno(confianza_final)
        
        # 3. Actualizar métricas
        self.actualizar_metricas(confianza_final)
        
        return resultado_atencion, confianza_final, alertas
    
    def actualizar_metricas(self, confianza):
        """Actualiza métricas de desempeño"""
        self.metricas_desempeno['total_predicciones'] += 1
        self.metricas_desempeno['confianza_promedio'] = (
            self.metricas_desempeno['confianza_promedio'] * 
            (self.metricas_desempeno['total_predicciones'] - 1) + confianza
        ) / self.metricas_desempeno['total_predicciones']
    
    def obtener_metricas(self):
        """Retorna métricas actuales"""
        return self.metricas_desempeno.copy()
    
    def evolucionar_reglas(self):
        """Aplica evolución a las reglas"""
        if self.entrenado and self.sistema_original.reglas_discriminantes:
            nuevas_reglas = self.evolucionador.evolucionar_reglas(
                self.sistema_original.reglas_discriminantes
            )
            self.sistema_original.reglas_discriminantes = nuevas_reglas
            return len(nuevas_reglas)
        return 0

# FUNCIONES PARA VISUALIZACIÓN DE MATRICES MEJORADAS
def visualizar_matrices_incidencias(sistema, grupo_maestro):
    """Visualiza las matrices de incidencias del grupo objetivo y contraste"""
    
    if sistema.sistema_original.matriz_maestro is None or sistema.sistema_original.matriz_contraste is None:
        st.warning("No hay matrices de incidencias disponibles. Primero entrena el sistema.")
        return
    
    st.header("📊 Visualización de Matrices de Incidencias")
    
    # Crear pestañas para diferentes visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Matrices Individuales", 
        "📈 Histogramas Básicos",
        "📊 Histogramas Avanzados", 
        "📋 Datos Numéricos",
        "📤 Exportar Datos"
    ])
    
    with tab1:
        st.subheader("Matrices de Transición de Polaridad")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Grupo Objetivo ({grupo_maestro})**")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(sistema.sistema_original.matriz_maestro, 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=GROUP_ORDER, 
                       yticklabels=GROUP_ORDER,
                       cbar_kws={'label': 'Frecuencia'})
            ax1.set_xlabel('Grupo Siguiente')
            ax1.set_ylabel('Grupo Actual')
            ax1.set_title(f'Matriz {grupo_maestro} - Total: {np.sum(sistema.sistema_original.matriz_maestro)}')
            st.pyplot(fig1)
        
        with col2:
            st.markdown("**Grupos de Contraste**")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.heatmap(sistema.sistema_original.matriz_contraste, 
                       annot=True, fmt='d', cmap='Reds',
                       xticklabels=GROUP_ORDER, 
                       yticklabels=GROUP_ORDER,
                       cbar_kws={'label': 'Frecuencia'})
            ax2.set_xlabel('Grupo Siguiente')
            ax2.set_ylabel('Grupo Actual')
            ax2.set_title(f'Matriz Contraste - Total: {np.sum(sistema.sistema_original.matriz_contraste)}')
            st.pyplot(fig2)
    
    with tab2:
        st.subheader("Histogramas Comparativos Básicos")
        
        # Preparar datos para histogramas
        datos_maestro = sistema.sistema_original.matriz_maestro.flatten()
        datos_contraste = sistema.sistema_original.matriz_contraste.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma grupo objetivo
        ax1.hist(datos_maestro, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Frecuencia de Transición')
        ax1.set_ylabel('Cantidad')
        ax1.set_title(f'Distribución {grupo_maestro}')
        ax1.grid(True, alpha=0.3)
        
        # Histograma grupos contraste
        ax2.hist(datos_contraste, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Frecuencia de Transición')
        ax2.set_ylabel('Cantidad')
        ax2.set_title('Distribución Grupos Contraste')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Histograma comparativo superpuesto
        st.subheader("Histograma Comparativo Superpuesto")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        ax3.hist(datos_maestro, bins=20, alpha=0.5, color='blue', label=grupo_maestro, edgecolor='black')
        ax3.hist(datos_contraste, bins=20, alpha=0.5, color='red', label='Contraste', edgecolor='black')
        ax3.set_xlabel('Frecuencia de Transición')
        ax3.set_ylabel('Cantidad')
        ax3.set_title('Comparación de Distribuciones')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
    
    with tab3:
        st.subheader("Histogramas Avanzados y Personalizados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Opciones de Personalización**")
            bins_maestro = st.slider("Número de bins para grupo objetivo:", 5, 50, 20, key="bins_maestro")
            bins_contraste = st.slider("Número de bins para grupo contraste:", 5, 50, 20, key="bins_contraste")
            densidad = st.checkbox("Mostrar como densidad de probabilidad", value=False)
            cumulative = st.checkbox("Mostrar histograma acumulativo", value=False)
        
        with col2:
            st.markdown("**Opciones de Visualización**")
            alpha_val = st.slider("Transparencia de histogramas:", 0.1, 1.0, 0.7)
            color_maestro = st.color_picker("Color grupo objetivo:", "#1f77b4")
            color_contraste = st.color_picker("Color grupo contraste:", "#d62728")
        
        # Crear histogramas personalizados
        fig_avanzado, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histograma personalizado grupo objetivo
        if densidad:
            ax1.hist(sistema.sistema_original.matriz_maestro.flatten(), bins=bins_maestro, 
                    alpha=alpha_val, color=color_maestro, edgecolor='black', density=True, cumulative=cumulative)
            ax1.set_ylabel('Densidad de Probabilidad')
        else:
            ax1.hist(sistema.sistema_original.matriz_maestro.flatten(), bins=bins_maestro, 
                    alpha=alpha_val, color=color_maestro, edgecolor='black', cumulative=cumulative)
            ax1.set_ylabel('Frecuencia')
        
        ax1.set_xlabel('Valor de Transición')
        ax1.set_title(f'Distribución {grupo_maestro} (Personalizado)')
        ax1.grid(True, alpha=0.3)
        
        # Histograma personalizado grupo contraste
        if densidad:
            ax2.hist(sistema.sistema_original.matriz_contraste.flatten(), bins=bins_contraste, 
                    alpha=alpha_val, color=color_contraste, edgecolor='black', density=True, cumulative=cumulative)
            ax2.set_ylabel('Densidad de Probabilidad')
        else:
            ax2.hist(sistema.sistema_original.matriz_contraste.flatten(), bins=bins_contraste, 
                    alpha=alpha_val, color=color_contraste, edgecolor='black', cumulative=cumulative)
            ax2.set_ylabel('Frecuencia')
        
        ax2.set_xlabel('Valor de Transición')
        ax2.set_title('Distribución Contraste (Personalizado)')
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig_avanzado)
        
        # Histograma de diferencias
        st.subheader("Histograma de Diferencias entre Grupos")
        
        # Calcular diferencias
        diferencias = sistema.sistema_original.matriz_maestro - sistema.sistema_original.matriz_contraste
        
        fig_diff, ax_diff = plt.subplots(figsize=(10, 6))
        im = ax_diff.imshow(diferencias, cmap='RdBu_r', interpolation='nearest')
        ax_diff.set_xlabel('Grupo Siguiente')
        ax_diff.set_ylabel('Grupo Actual')
        ax_diff.set_title('Diferencias entre Matrices (Objetivo - Contraste)')
        ax_diff.set_xticks(range(4))
        ax_diff.set_yticks(range(4))
        ax_diff.set_xticklabels(GROUP_ORDER)
        ax_diff.set_yticklabels(GROUP_ORDER)
        
        # Añadir anotaciones
        for i in range(4):
            for j in range(4):
                ax_diff.text(j, i, f'{diferencias[i, j]:+d}', 
                           ha="center", va="center", 
                           color="white" if abs(diferencias[i, j]) > np.max(np.abs(diferencias))/2 else "black")
        
        plt.colorbar(im, ax=ax_diff, label='Diferencia')
        st.pyplot(fig_diff)
    
    with tab4:
        st.subheader("Datos Numéricos Detallados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Matriz {grupo_maestro}**")
            df_maestro = pd.DataFrame(
                sistema.sistema_original.matriz_maestro,
                index=GROUP_ORDER,
                columns=GROUP_ORDER
            )
            st.dataframe(df_maestro.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # Estadísticas adicionales
            st.markdown("**Estadísticas del Grupo Objetivo:**")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Total transiciones", np.sum(sistema.sistema_original.matriz_maestro))
            with col_stats2:
                st.metric("Transición más frecuente", 
                         f"{GROUP_ORDER[np.unravel_index(np.argmax(sistema.sistema_original.matriz_maestro), sistema.sistema_original.matriz_maestro.shape)[0]]}→"
                         f"{GROUP_ORDER[np.unravel_index(np.argmax(sistema.sistema_original.matriz_maestro), sistema.sistema_original.matriz_maestro.shape)[1]]}")
            with col_stats3:
                st.metric("Valor máximo", np.max(sistema.sistema_original.matriz_maestro))
        
        with col2:
            st.markdown("**Matriz Contraste**")
            df_contraste = pd.DataFrame(
                sistema.sistema_original.matriz_contraste,
                index=GROUP_ORDER,
                columns=GROUP_ORDER
            )
            st.dataframe(df_contraste.style.background_gradient(cmap='Reds'), use_container_width=True)
            
            # Estadísticas adicionales
            st.markdown("**Estadísticas del Grupo Contraste:**")
            col_stats4, col_stats5, col_stats6 = st.columns(3)
            with col_stats4:
                st.metric("Total transiciones", np.sum(sistema.sistema_original.matriz_contraste))
            with col_stats5:
                st.metric("Transición más frecuente", 
                         f"{GROUP_ORDER[np.unravel_index(np.argmax(sistema.sistema_original.matriz_contraste), sistema.sistema_original.matriz_contraste.shape)[0]]}→"
                         f"{GROUP_ORDER[np.unravel_index(np.argmax(sistema.sistema_original.matriz_contraste), sistema.sistema_original.matriz_contraste.shape)[1]]}")
            with col_stats6:
                st.metric("Valor máximo", np.max(sistema.sistema_original.matriz_contraste))
        
        # Matriz de diferencias - CORREGIDO: sin parámetro center
        st.subheader("Matriz de Diferencias")
        diferencias = sistema.sistema_original.matriz_maestro - sistema.sistema_original.matriz_contraste
        df_diferencias = pd.DataFrame(
            diferencias,
            index=GROUP_ORDER,
            columns=GROUP_ORDER
        )
        
        # Usar vmin y vmax para centrar el gradiente en 0
        vmin = np.min(diferencias)
        vmax = np.max(diferencias)
        abs_max = max(abs(vmin), abs(vmax))
        
        st.dataframe(df_diferencias.style.background_gradient(cmap='RdBu_r', vmin=-abs_max, vmax=abs_max), 
                    use_container_width=True)
        
        # Resumen de diferencias
        st.markdown("**Resumen de Diferencias:**")
        col_diff1, col_diff2, col_diff3 = st.columns(3)
        with col_diff1:
            st.metric("Diferencia máxima", f"+{np.max(diferencias)}")
        with col_diff2:
            st.metric("Diferencia mínima", f"{np.min(diferencias)}")
        with col_diff3:
            st.metric("Diferencia promedio", f"{np.mean(diferencias):.2f}")
    
    with tab5:
        st.subheader("Exportar Datos de Matrices")
        
        # Crear DataFrames para exportación
        df_maestro_export = pd.DataFrame(
            sistema.sistema_original.matriz_maestro,
            index=[f"From_{group}" for group in GROUP_ORDER],
            columns=[f"To_{group}" for group in GROUP_ORDER]
        )
        
        df_contraste_export = pd.DataFrame(
            sistema.sistema_original.matriz_contraste,
            index=[f"From_{group}" for group in GROUP_ORDER],
            columns=[f"To_{group}" for group in GROUP_ORDER]
        )
        
        # Matriz de diferencias para exportación
        diferencias = sistema.sistema_original.matriz_maestro - sistema.sistema_original.matriz_contraste
        df_diferencias_export = pd.DataFrame(
            diferencias,
            index=[f"From_{group}" for group in GROUP_ORDER],
            columns=[f"To_{group}" for group in GROUP_ORDER]
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Matriz Objetivo**")
            st.download_button(
                label="📥 Descargar CSV",
                data=df_maestro_export.to_csv().encode('utf-8'),
                file_name=f"matriz_objetivo_{grupo_maestro}.csv",
                mime="text/csv",
                help="Descargar matriz del grupo objetivo en formato CSV"
            )
            
            # Datos para histograma
            datos_maestro_export = pd.DataFrame({
                'valor': sistema.sistema_original.matriz_maestro.flatten(),
                'grupo': grupo_maestro
            })
            st.download_button(
                label="📊 Datos Histograma",
                data=datos_maestro_export.to_csv().encode('utf-8'),
                file_name=f"datos_histograma_{grupo_maestro}.csv",
                mime="text/csv",
                help="Descargar datos para histograma del grupo objetivo"
            )
        
        with col2:
            st.markdown("**Matriz Contraste**")
            st.download_button(
                label="📥 Descargar CSV",
                data=df_contraste_export.to_csv().encode('utf-8'),
                file_name="matriz_contraste.csv",
                mime="text/csv",
                help="Descargar matriz de grupos contraste en formato CSV"
            )
            
            # Datos para histograma
            datos_contraste_export = pd.DataFrame({
                'valor': sistema.sistema_original.matriz_contraste.flatten(),
                'grupo': 'contraste'
            })
            st.download_button(
                label="📊 Datos Histograma",
                data=datos_contraste_export.to_csv().encode('utf-8'),
                file_name="datos_histograma_contraste.csv",
                mime="text/csv",
                help="Descargar datos para histograma del grupo contraste"
            )
        
        with col3:
            st.markdown("**Matriz Diferencias**")
            st.download_button(
                label="📥 Descargar CSV",
                data=df_diferencias_export.to_csv().encode('utf-8'),
                file_name="matriz_diferencias.csv",
                mime="text/csv",
                help="Descargar matriz de diferencias en formato CSV"
            )
            
            # Datos combinados para histograma comparativo
            datos_combinados = pd.DataFrame({
                'valor': np.concatenate([
                    sistema.sistema_original.matriz_maestro.flatten(),
                    sistema.sistema_original.matriz_contraste.flatten()
                ]),
                'grupo': [grupo_maestro] * len(sistema.sistema_original.matriz_maestro.flatten()) + 
                        ['contraste'] * len(sistema.sistema_original.matriz_contraste.flatten())
            })
            st.download_button(
                label="📈 Datos Combinados",
                data=datos_combinados.to_csv().encode('utf-8'),
                file_name="datos_combinados_histogramas.csv",
                mime="text/csv",
                help="Descargar datos combinados para histogramas comparativos"
            )
        
        # Exportar imágenes
        st.subheader("Exportar Visualizaciones")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Crear figura para exportación de matrices
            fig_export, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Matriz objetivo
            sns.heatmap(sistema.sistema_original.matriz_maestro, 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=GROUP_ORDER, 
                       yticklabels=GROUP_ORDER,
                       ax=ax1, cbar_kws={'label': 'Frecuencia'})
            ax1.set_title(f'Matriz {grupo_maestro}')
            ax1.set_xlabel('Grupo Siguiente')
            ax1.set_ylabel('Grupo Actual')
            
            # Matriz contraste
            sns.heatmap(sistema.sistema_original.matriz_contraste, 
                       annot=True, fmt='d', cmap='Reds',
                       xticklabels=GROUP_ORDER, 
                       yticklabels=GROUP_ORDER,
                       ax=ax2, cbar_kws={'label': 'Frecuencia'})
            ax2.set_title('Matriz Contraste')
            ax2.set_xlabel('Grupo Siguiente')
            ax2.set_ylabel('Grupo Actual')
            
            plt.tight_layout()
            
            # Convertir a bytes para descarga
            from io import BytesIO
            buf_matrices = BytesIO()
            fig_export.savefig(buf_matrices, format="png", dpi=300, bbox_inches='tight')
            buf_matrices.seek(0)
            
            st.download_button(
                label="🖼️ Matrices (PNG)",
                data=buf_matrices,
                file_name="matrices_comparativas.png",
                mime="image/png",
                help="Descargar visualización de ambas matrices en PNG"
            )
        
        with col5:
            # Crear figura de histogramas básicos para exportación
            fig_hist_basico, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            datos_maestro = sistema.sistema_original.matriz_maestro.flatten()
            datos_contraste = sistema.sistema_original.matriz_contraste.flatten()
            
            ax1.hist(datos_maestro, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title(f'Distribución {grupo_maestro}')
            ax1.set_xlabel('Frecuencia')
            ax1.set_ylabel('Cantidad')
            ax1.grid(True, alpha=0.3)
            
            ax2.hist(datos_contraste, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax2.set_title('Distribución Contraste')
            ax2.set_xlabel('Frecuencia')
            ax2.set_ylabel('Cantidad')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convertir a bytes para descarga
            buf_hist_basico = BytesIO()
            fig_hist_basico.savefig(buf_hist_basico, format="png", dpi=300, bbox_inches='tight')
            buf_hist_basico.seek(0)
            
            st.download_button(
                label="📊 Histogramas Básicos (PNG)",
                data=buf_hist_basico,
                file_name="histogramas_basicos.png",
                mime="image/png",
                help="Descargar histogramas básicos en PNG"
            )
        
        with col6:
            # Crear figura de histograma comparativo para exportación
            fig_hist_comp, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(datos_maestro, bins=20, alpha=0.5, color='blue', label=grupo_maestro, edgecolor='black')
            ax.hist(datos_contraste, bins=20, alpha=0.5, color='red', label='Contraste', edgecolor='black')
            ax.set_xlabel('Frecuencia de Transición')
            ax.set_ylabel('Cantidad')
            ax.set_title('Comparación de Distribuciones')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convertir a bytes para descarga
            buf_hist_comp = BytesIO()
            fig_hist_comp.savefig(buf_hist_comp, format="png", dpi=300, bbox_inches='tight')
            buf_hist_comp.seek(0)
            
            st.download_button(
                label="📈 Histograma Comparativo (PNG)",
                data=buf_hist_comp,
                file_name="histograma_comparativo.png",
                mime="image/png",
                help="Descargar histograma comparativo en PNG"
            )

def main():
    st.title("🧬 Polarity Index Method (PIM)")
    st.markdown("""
    ### Todas las Mejoras Implementadas + Seguridad + Subida de Archivos + Visualización de Matrices
    **Sistema completo con IA regenerativa, medidas de seguridad y análisis visual**
    """)
    
    # Inicializar sistema en session state
    if 'sistema_evolutivo' not in st.session_state:
        st.session_state.sistema_evolutivo = PolaridadEvolutiva()
    
    sistema = st.session_state.sistema_evolutivo
    
    # Sidebar con información y controles
    with st.sidebar:
        st.header("⚙️ Sistema Evolutivo Avanzado")
        st.info("""
        **Mejoras Implementadas:**
        - ✅ Sistema híbrido completo
        - ✅ Ensemble de perfiles  
        - ✅ Mecanismo de atención
        - ✅ Auto-evolución de reglas
        - ✅ Patrones compuestos
        - ✅ Sistema de alertas
        - 🔐 Medidas de seguridad
        - 📁 Subida de archivos
        - 📊 Visualización de matrices
        - 📈 Histogramas avanzados
        """)
        
        # Estado del sistema
        st.header("🔍 Estado del Sistema")
        if sistema.entrenado:
            st.success("✅ Sistema evolutivo entrenado")
            metricas = sistema.obtener_metricas()
            st.metric("Predicciones", metricas['total_predicciones'])
            st.metric("Confianza promedio", f"{metricas['confianza_promedio']:.1f}%")
            
            # Botón para evolución manual
            if st.button("🔄 Evolucionar Reglas", type="secondary"):
                nuevas_reglas = sistema.evolucionar_reglas()
                st.success(f"Reglas evolucionadas: {nuevas_reglas} reglas activas")
        else:
            st.error("❌ Sistema no entrenado")
        
        st.header("📊 Estadísticas")
        if sistema.entrenado:
            st.write(f"**Hash del modelo:** `{sistema.modelo_hash}`")
            st.write(f"**Reglas activas:** {len(sistema.sistema_original.reglas_discriminantes)}")
            st.write(f"**Alertas activas:** {len(sistema.sistema_alertas.alertas_activas)}")
    
    # 🔐 SECCIÓN DE SUBIDA DE ARCHIVOS
    st.header("📁 Cargar Archivos FASTA")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.subheader("Archivo Objetivo (Grupo Maestro)")
        archivo_objetivo = st.file_uploader(
            "Subir archivo FASTA del grupo objetivo",
            type=['fasta', 'fa', 'txt'],
            key="objetivo_uploader"
        )
        
        if archivo_objetivo:
            try:
                SecurityConfig.validate_file_upload(archivo_objetivo)
                contenido_objetivo = archivo_objetivo.getvalue().decode('utf-8')
                st.success(f"✅ Archivo objetivo cargado: {archivo_objetivo.name}")
                st.session_state.contenido_objetivo = contenido_objetivo
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col_arch2:
        st.subheader("Archivos de Contraste")
        archivos_contraste = st.file_uploader(
            "Subir archivos FASTA de grupos de contraste",
            type=['fasta', 'fa', 'txt'],
            accept_multiple_files=True,
            key="contraste_uploader"
        )
        
        if archivos_contraste:
            contenido_contraste = ""
            for archivo in archivos_contraste:
                try:
                    SecurityConfig.validate_file_upload(archivo)
                    contenido_contraste += archivo.getvalue().decode('utf-8') + "\n"
                    st.success(f"✅ {archivo.name} cargado")
                except Exception as e:
                    st.error(f"❌ Error en {archivo.name}: {e}")
            
            if contenido_contraste:
                st.session_state.contenido_contraste = contenido_contraste
    
    # Entrada de datos manual como alternativa
    st.header("📝 O Ingresar Datos Manualmente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grupo_maestro = st.text_input("Grupo Maestro (ej: virus):", "virus")
        if 'contenido_objetivo' in st.session_state:
            fasta_maestro = st.session_state.contenido_objetivo
        else:
            fasta_maestro = st.text_area(
                f"Secuencias FASTA del grupo {grupo_maestro}:",
                height=150,
                placeholder=">proteina_virus_1\nMKTIIALSYIFCL...\n>proteina_virus_2\nMKAL..."
            )
    
    with col2:
        grupos_contraste = st.text_input("Grupos de contraste (separados por coma):", "bacteria,humano,planta")
        if 'contenido_contraste' in st.session_state:
            fasta_contraste = st.session_state.contenido_contraste
        else:
            fasta_contraste = st.text_area(
                "Secuencias FASTA de grupos de contraste:",
                height=150,
                placeholder=">proteina_bacteria_1\nMKKLW...\n>proteina_humano_1\nMALSL..."
            )
    
    # Datos de prueba pre-cargados
    with st.expander("🧪 Cargar datos de prueba predefinidos"):
        if st.button("Cargar ejemplos virus vs otros"):
            st.session_state.fasta_maestro = """>proteina_virus_1
MKTIIALSYIFCLVFADYKDDDDKHHHHHH
>proteina_virus_2  
MKALSLALSLALSLAHHHHKKKRRR
>proteina_virus_3
MKKLWGLALSLALSLALSLSDDEE
>proteina_virus_4
MRRRHHHKKKSSSTTTNNOO
>proteina_virus_5
MAAAVVVIILLLPPPGGRRKK"""

            st.session_state.fasta_contraste = """>proteina_bacteria_1
MSSSSSTTTTNNQQQCCYYYWWW
>proteina_bacteria_2
MDDDDEEEEAAAVVVIILLL
>proteina_humano_1
MRRRKKKHHHDDEEESSSTTT
>proteina_humano_2
MAAASSSDDDFFFGGGHHHKKK
>proteina_planta_1
MNNNQQQCCCYYYWWWTTTSSS
>proteina_planta_2
MVVVIILLLLPPPGGGAAASS"""
            
            st.rerun()
    
    # Usar datos de sesión si existen
    if 'fasta_maestro' in st.session_state:
        fasta_maestro = st.session_state.fasta_maestro
    if 'fasta_contraste' in st.session_state:
        fasta_contraste = st.session_state.fasta_contraste
    
    # BOTÓN PRINCIPAL DE ENTRENAMIENTO
    if st.button("🚀 Entrenar Sistema Evolutivo Completo", type="primary", key="entrenar_evolutivo"):
        if not fasta_maestro or not fasta_contraste:
            st.error("Debe proporcionar secuencias para ambos grupos")
        else:
            try:
                # Procesar secuencias
                secuencias_maestro = sistema.sistema_original.parse_fasta(fasta_maestro)
                secuencias_contraste = sistema.sistema_original.parse_fasta(fasta_contraste)
                
                if not secuencias_maestro or not secuencias_contraste:
                    st.error("No se pudieron procesar las secuencias FASTA")
                    return
                
                # Mostrar estadísticas de datos
                st.info(f"📊 Datos cargados: {len(secuencias_maestro)} secuencias objetivo, {len(secuencias_contraste)} secuencias contraste")
                
                # Entrenamiento completo
                with st.status("🏗️ Entrenando sistema evolutivo completo...", expanded=True) as status:
                    status.write("🔍 Procesando secuencias...")
                    resultados = sistema.entrenar_sistema_completo(
                        secuencias_maestro, secuencias_contraste, grupo_maestro
                    )
                    
                    status.write("✅ Sistema base entrenado")
                    status.write("🧠 Configurando componentes de IA...")
                    status.write("🔬 Analizando patrones avanzados...")
                
                # Mostrar resultados completos
                st.header("🎉 Entrenamiento Evolutivo Completado")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Reglas Maestro", resultados['reglas_maestro'])
                with col2:
                    st.metric("Reglas Otros", resultados['reglas_otros'])
                with col3:
                    st.metric("Discriminantes", resultados['discriminantes'])
                with col4:
                    st.metric("Posiciones Clave", resultados['posiciones_importantes'])
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Patrones Compuestos", resultados['patrones_compuestos'])
                with col6:
                    st.metric("Hash Modelo", resultados['modelo_hash'][:8])
                with col7:
                    st.metric("Timestamp", resultados['timestamp'][11:])
                
                st.success("✅ Sistema evolutivo avanzado listo para clasificar")
                
            except Exception as e:
                st.error(f"❌ Error en el entrenamiento: {str(e)}")
                import traceback
                with st.expander("🔍 Ver detalles del error"):
                    st.code(traceback.format_exc())

    # SECCIÓN DE VISUALIZACIÓN DE MATRICES
    if sistema.entrenado:
        st.header("📊 Visualización de Matrices de Incidencias")
        visualizar_matrices_incidencias(sistema, grupo_maestro)

    # SECCIÓN DE PRUEBA AVANZADA
    if sistema.entrenado:
        st.header("🧪 Probar Sistema Evolutivo Avanzado")
        
        # Mostrar alertas si existen
        if sistema.sistema_alertas.alertas_activas:
            st.warning("🚨 Alertas Activas del Sistema")
            for alerta in sistema.sistema_alertas.alertas_activas:
                st.error(f"{alerta['tipo']}: {alerta['mensaje']}")
        
        col_test1, col_test2 = st.columns([3, 1])
        
        with col_test1:
            proteina_prueba = st.text_input(
                "Secuencia de prueba:",
                "MKTIIALSYIFCLVFADYK",
                key="secuencia_prueba_evolutivo"
            )
        
        with col_test2:
            st.write("")  # Espacio vertical
            st.write("")  # Espacio vertical
            boton_clasificar = st.button("🔍 Clasificar con Sistema Avanzado", 
                                       type="secondary", 
                                       key="clasificar_avanzado")
        
        if boton_clasificar:
            if proteina_prueba:
                with st.spinner("Clasificando con sistema avanzado..."):
                    resultado, confianza, alertas = sistema.clasificar_evolutivo(proteina_prueba)
                
                # Mostrar resultados comparativos avanzados
                st.subheader("📊 Resultados del Sistema Avanzado")
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.write("**Sistema Evolutivo Avanzado:**")
                    if resultado == "Yes":
                        st.success(f"**Resultado:** {resultado}")
                    else:
                        st.error(f"**Resultado:** {resultado}")
                    st.metric("Confianza", f"{confianza}%")
                    
                    # Mostrar componentes de la decisión
                    with st.expander("🔍 Componentes de la decisión"):
                        resultado_orig, confianza_orig = sistema.sistema_original.evaluar_proteina_original(proteina_prueba)
                        st.write(f"**Sistema base:** {confianza_orig}%")
                        st.write(f"**Patrones compuestos:** +{int(sistema.detector_patrones.evaluar_patron_compuesto(proteina_prueba) * 20)}%")
                        st.write(f"**Atención posicional:** Ajuste aplicado")
                
                with col_comp2:
                    # Resultado del sistema original para comparar
                    resultado_orig, confianza_orig = sistema.sistema_original.evaluar_proteina_original(proteina_prueba)
                    st.write("**Sistema Original:**")
                    if resultado_orig == "Yes":
                        st.success(f"**Resultado:** {resultado_orig}")
                    else:
                        st.error(f"**Resultado:** {resultado_orig}")
                    st.metric("Confianza", f"{confianza_orig}%")
                
                with col_comp3:
                    # Mejora calculada
                    mejora = confianza - confianza_orig
                    st.write("**Mejora del Sistema:**")
                    if mejora > 0:
                        st.success(f"**+{mejora}%** de mejora")
                    else:
                        st.error(f"**{mejora}%** de mejora")
                    st.metric("Confianza Promedio", f"{sistema.metricas_desempeno['confianza_promedio']:.1f}%")
                
                # Interpretación avanzada
                st.subheader("🎯 Interpretación Avanzada")
                
                if resultado == "Yes":
                    st.success(f"✅ La proteína es clasificada como del grupo **{grupo_maestro}**")
                    
                    # Análisis detallado
                    with st.expander("🔬 Análisis Detallado"):
                        st.write("**Características identificadas:**")
                        
                        # Patrones compuestos encontrados
                        score_patrones = sistema.detector_patrones.evaluar_patron_compuesto(proteina_prueba)
                        if score_patrones > 0.3:
                            st.write(f"✅ Patrones compuestos coincidentes: {score_patrones:.1%}")
                        
                        # Posiciones importantes
                        lineal = sistema.sistema_original.secuencia_a_lineal(proteina_prueba)
                        pos_importantes = [i for i in range(16) if i in sistema.sistema_atencion.posiciones_importantes and lineal[i] > 0]
                        if pos_importantes:
                            st.write(f"✅ Posiciones clave activas: {len(pos_importantes)}")
                        
                        # Reglas aplicadas
                        reglas_aplicadas = len([r for r in sistema.sistema_original.reglas_discriminantes[:5] 
                                              if lineal[r['posicion']-1] == r['valor']])
                        st.write(f"✅ Reglas discriminantes: {5 - reglas_aplicadas}/5 pasadas")
                    
                    st.balloons()
                else:
                    st.error(f"❌ La proteína NO es del grupo **{grupo_maestro}**")
                    
                    with st.expander("🔍 Razones de rechazo"):
                        lineal = sistema.sistema_original.secuencia_a_lineal(proteina_prueba)
                        reglas_falladas = [r for r in sistema.sistema_original.reglas_discriminantes[:5] 
                                         if lineal[r['posicion']-1] == r['valor']]
                        if reglas_falladas:
                            st.write(f"❌ Reglas discriminantes falladas: {len(reglas_falladas)}")
                    
            else:
                st.warning("Por favor, ingresa una secuencia para clasificar")
    else:
        st.warning("⚠️ Primero debes entrenar el sistema evolutivo")

    # INFORMACIÓN COMPLETA SOBRE MEJORAS
    with st.expander("🏆 Resumen de Todas las Mejoras Implementadas"):
        st.markdown("""
        ## 🧠 MEJORAS DE IA REGENERATIVA Y ML
        
        ### **Mejora 1: Sistema Híbrido Evolutivo**
        - Ensemble que combina múltiples perspectivas
        - Arquitectura extensible para futuras mejoras
        
        ### **Mejora 2: Mecanismo de Atención**  
        - Identifica posiciones clave en secuencias
        - Ajusta confianza basado en importancia posicional
        
        ### **Mejora 3: Auto-evolución de Reglas**
        - Las reglas inefectivas se mutan automáticamente
        - Las reglas efectivas se preservan y fortalecen
        
        ### **Mejora 4: Detección de Patrones Compuestos**
        - Identifica patrones que involucran múltiples posiciones
        - Añade capa adicional de análisis estructural
        
        ### **Mejora 5: Sistema de Alertas Tempranas**
        - Detecta deriva de concepto en tiempo real
        - Monitorea desempeño continuo del sistema
        
        ### **Mejora 6: Medidas de Seguridad**
        - Validación de archivos subidos
        - Límites de tamaño y secuencias
        - Hash único para cada modelo
        
        ### **Mejora 7: Subida de Archivos FASTA**
        - Carga múltiples archivos de contraste
        - Validación de formato y seguridad
        - Compatibilidad con entrada manual
        
        ### **Mejora 8: Visualización de Matrices de Incidencias**
        - Matrices de calor para grupos objetivo y contraste
        - Histogramas comparativos de distribuciones
        - Exportación de datos en CSV y PNG
        - Análisis visual de patrones de transición
        
        ### **Mejora 9: Histogramas Avanzados**
        - Personalización de bins y colores
        - Histogramas de densidad y acumulativos
        - Visualización de diferencias entre grupos
        - Exportación de datos para análisis externo
        """)

if __name__ == "__main__":
    main()
