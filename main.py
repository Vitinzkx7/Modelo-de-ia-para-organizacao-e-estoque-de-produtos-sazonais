import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import mysql.connector
from dotenv import load_dotenv
# Modelagem e Previsão
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Séries Temporais
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# Otimização
from scipy.optimize import minimize
import pulp

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# Carregando Variáveis de Ambiente
load_dotenv()

# Conexão do Banco de Dados
try:
    # Estabelece a conexão usando as variáveis de ambiente
    conexao = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

    if conexao.is_connected():
        print("Conexão bem-sucedida ao MySQL!")

except mysql.connector.Error as err:
    print(f"Erro: {err}")

finally:
    if 'conexao' in locals() and conexao.is_connected():
        conexao.close()

# CLASSE PRINCIPAL


class PrevisaoDemandaSazonal:
    """
    Classe principal para previsão de demanda e otimização de estoque
    para produtos com padrões sazonais
    """

    def __init__(self, nome_produto="Produto_Sazonal"):
        """
        Inicializa o modelo de previsão

        Args:
            nome_produto (str): Nome do produto para identificação
        """
        self.nome_produto = nome_produto
        self.modelo_serie_temporal = None
        self.modelo_machine_learning = None
        self.dados = None
        self.previsoes = None
        self.metricas = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

        print(f" Sistema inicializado para: {nome_produto}")


    #  GERAÇÃO DE DADOS SINTÉTICOS


    def gerar_dados_sinteticos(self, periodo_dias=1095,
                               tendencia_base=100,
                               sazonalidade_anual=30,
                               ruido=15,
                               eventos_especiais=True):
        """
        Gera dados sintéticos realistas para produtos sazonais

        Args:
            periodo_dias: Número de dias para gerar dados
            tendencia_base: Demanda base média
            sazonalidade_anual: Amplitude da sazonalidade anual
            ruido: Nível de ruído aleatório
            eventos_especiais: Incluir eventos especiais (feriados, promoções)
        """

        print(" Gerando dados sintéticos...")

        # Datas
        datas = pd.date_range(end=datetime.now(), periods=periodo_dias, freq='D')

        # Componentes da demanda
        t = np.arange(len(datas))

        # Tendência de longo prazo (crescimento suave)
        tendencia = tendencia_base + 0.02 * t

        # Sazonalidade anual
        sazonalidade_anual = sazonalidade_anual * np.sin(2 * np.pi * t / 365)

        # Sazonalidade semanal
        dia_semana = datas.dayofweek
        sazonalidade_semanal = np.where(dia_semana < 5, 10, 20)  # Fim de semana tem mais demanda
        sazonalidade_semanal = sazonalidade_semanal * np.sin(2 * np.pi * t / 7)

        # Eventos especiais
        eventos = np.zeros(len(datas))

        if eventos_especiais:
            # Natal (25/dez)
            natal_mask = (datas.month == 12) & (datas.day == 25)
            eventos[natal_mask] = 50

            # Ano Novo (1/jan)
            ano_novo_mask = (datas.month == 1) & (datas.day == 1)
            eventos[ano_novo_mask] = 40

            # Black Friday (última sexta de novembro)
            for ano in datas.year.unique():
                novembro = pd.date_range(f'{ano}-11-01', f'{ano}-11-30')
                black_friday = novembro[novembro.dayofweek == 4][-1]  # Última sexta
                if black_friday in datas:
                    idx = np.where(datas == black_friday)[0]
                    eventos[idx] = 80
                    # Efeito nos dias próximos
                    for i in range(-3, 4):
                        if idx + i < len(datas) and idx + i >= 0:
                            eventos[idx + i] += 30 * (1 - abs(i) / 4)

            # Promoções aleatórias
            num_promocoes = periodo_dias // 30
            for _ in range(num_promocoes):
                dia_promo = np.random.randint(30, periodo_dias - 30)
                eventos[dia_promo:dia_promo + 7] += 25

        # Ruído aleatório
        ruido_aleatorio = np.random.normal(0, ruido, len(datas))

        # Demanda final
        demanda = (tendencia +
                   sazonalidade_anual +
                   sazonalidade_semanal +
                   eventos +
                   ruido_aleatorio)

        # Garantir valores não negativos
        demanda = np.maximum(demanda, 0)

        # Criar DataFrame
        self.dados = pd.DataFrame({
            'data': datas,
            'demanda': demanda,
            'preco': 100 + 10 * np.sin(2 * np.pi * t / 180) + np.random.normal(0, 2, len(datas)),
            'temperatura': 20 + 10 * np.sin(2 * np.pi * t / 365 - np.pi / 2) + np.random.normal(0, 3, len(datas)),
            'feriado': ((datas.month == 12) & (datas.day == 25)) |
                       ((datas.month == 1) & (datas.day == 1)),
            'fim_semana': datas.dayofweek >= 5,
            'mes': datas.month,
            'ano': datas.year,
            'dia_semana': datas.dayofweek,
            'trimestre': datas.quarter
        })

        # Adicionar features temporais
        self._adicionar_features_temporais()

        print(f" Dados gerados: {len(self.dados)} registros")
        print(f"   Período: {datas[0].strftime('%Y-%m-%d')} a {datas[-1].strftime('%Y-%m-%d')}")
        print(f"   Demanda média: {demanda.mean():.1f}")
        print(f"   Demanda máxima: {demanda.max():.1f}")
        print(f"   Demanda mínima: {demanda.min():.1f}")

        return self.dados

    def _adicionar_features_temporais(self):
        """Adiciona features temporais derivadas"""
        if self.dados is None:
            return

        # Features de lag
        for lag in [1, 2, 3, 7, 14, 30]:
            self.dados[f'lag_{lag}'] = self.dados['demanda'].shift(lag)

        # Médias móveis
        for window in [7, 14, 30]:
            self.dados[f'media_movel_{window}'] = self.dados['demanda'].rolling(window=window).mean()

        # Features de data
        self.dados['dia_do_ano'] = self.dados['data'].dt.dayofyear
        self.dados['semana_do_ano'] = self.dados['data'].dt.isocalendar().week
        self.dados['estacao'] = (self.dados['mes'] % 12 + 3) // 3  # 1: Verão, 2: Outono, 3: Inverno, 4: Primavera


    #  ANÁLISE EXPLORATÓRIA


    def analisar_dados(self):
        """Realiza análise exploratória completa dos dados"""

        if self.dados is None:
            print(" Nenhum dado disponível. Execute gerar_dados_sinteticos() primeiro.")
            return

        print("\n" + "=" * 60)
        print(" ANÁLISE EXPLORATÓRIA DOS DADOS")
        print("=" * 60)

        # Estatísticas descritivas
        print("\n Estatísticas Descritivas da Demanda:")
        print(self.dados['demanda'].describe())

        # Teste de estacionariedade
        resultado_adf = adfuller(self.dados['demanda'].dropna())
        print(f"\n Teste de Dickey-Fuller Aumentado:")
        print(f"   Estatística ADF: {resultado_adf[0]:.4f}")
        print(f"   p-valor: {resultado_adf[1]:.4f}")
        print(f"   Estacionário: {'Sim' if resultado_adf[1] < 0.05 else 'Não'}")

        # Decomposição sazonal
        self._decompor_serie_temporal()

        # Análise de correlações
        self._analisar_correlacoes()

        # Visualizações
        self._plotar_analises()

    def _decompor_serie_temporal(self):
        """Decompõe a série temporal em tendência, sazonalidade e resíduo"""

        # Usar dados mensais para decomposição
        dados_mensais = self.dados.set_index('data')['demanda'].resample('M').mean()

        if len(dados_mensais) >= 24:  # Precisamos de pelo menos 2 anos
            decomposicao = seasonal_decompose(dados_mensais, model='additive', period=12)

            print("\n Decomposição da Série Temporal:")
            print(
                f"   Força da Tendência: {1 - np.var(decomposicao.resid) / np.var(decomposicao.trend + decomposicao.resid):.3f}")
            print(
                f"   Força da Sazonalidade: {1 - np.var(decomposicao.resid) / np.var(decomposicao.seasonal + decomposicao.resid):.3f}")

            return decomposicao

    def _analisar_correlacoes(self):
        """Analisa correlações entre variáveis"""

        # Selecionar colunas numéricas
        colunas_numericas = self.dados.select_dtypes(include=[np.number]).columns
        correlacoes = self.dados[colunas_numericas].corr()['demanda'].sort_values(ascending=False)

        print("\n Top 10 Correlações com Demanda:")
        for var, corr in correlacoes[1:11].items():
            print(f"   {var}: {corr:.3f}")

    def _plotar_analises(self):
        """Cria visualizações dos dados"""

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Análise de Demanda - {self.nome_produto}', fontsize=16)

        #  Série temporal completa
        axes[0, 0].plot(self.dados['data'], self.dados['demanda'], alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Demanda Diária')
        axes[0, 0].set_xlabel('Data')
        axes[0, 0].set_ylabel('Demanda')
        axes[0, 0].tick_params(axis='x', rotation=45)

        #  Distribuição da demanda
        axes[0, 1].hist(self.dados['demanda'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribuição da Demanda')
        axes[0, 1].set_xlabel('Demanda')
        axes[0, 1].set_ylabel('Frequência')

        #  Demanda por mês
        self.dados.boxplot(column='demanda', by='mes', ax=axes[1, 0])
        axes[1, 0].set_title('Demanda por Mês')
        axes[1, 0].set_xlabel('Mês')
        axes[1, 0].set_ylabel('Demanda')

        #  Demanda por dia da semana
        dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        demanda_dia = self.dados.groupby('dia_semana')['demanda'].mean()
        axes[1, 1].bar(dias, demanda_dia)
        axes[1, 1].set_title('Demanda Média por Dia da Semana')
        axes[1, 1].set_ylabel('Demanda Média')

        #  Sazonalidade anual
        demanda_media_anual = self.dados.groupby('dia_do_ano')['demanda'].mean()
        axes[2, 0].plot(demanda_media_anual.index, demanda_media_anual.values)
        axes[2, 0].set_title('Padrão Sazonal Anual')
        axes[2, 0].set_xlabel('Dia do Ano')
        axes[2, 0].set_ylabel('Demanda Média')

        #  Correlações
        colunas_corr = ['demanda', 'preco', 'temperatura', 'lag_1', 'lag_7', 'media_movel_7']
        colunas_existentes = [c for c in colunas_corr if c in self.dados.columns]
        if len(colunas_existentes) > 1:
            corr_matrix = self.dados[colunas_existentes].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        ax=axes[2, 1], fmt='.2f', square=True)
            axes[2, 1].set_title('Matriz de Correlação')

        plt.tight_layout()
        plt.show()


    # MODELOS DE PREVISÃO


    def preparar_dados_ml(self, teste_tamanho=0.2):
        """
        Preparar dados para modelos de machine learning

        Args:
            teste_tamanho: Proporção dos dados para teste
        """

        if self.dados is None:
            print("❌ Nenhum dado disponível.")
            return None, None, None, None

        # Remover linhas com NaN (devido aos lags)
        dados_ml = self.dados.dropna().copy()

        # Features para o modelo
        features = ['preco', 'temperatura', 'feriado', 'fim_semana',
                    'mes', 'dia_semana', 'trimestre', 'lag_1', 'lag_7',
                    'lag_14', 'lag_30', 'media_movel_7', 'media_movel_14']

        # Selecionar apenas features disponíveis
        features_disponiveis = [f for f in features if f in dados_ml.columns]

        X = dados_ml[features_disponiveis]
        y = dados_ml['demanda']

        # Dividir em treino e teste (mantendo ordem temporal)
        split_idx = int(len(X) * (1 - teste_tamanho))

        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]

        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Salvar datas para referência
        self.dates_train = dados_ml['data'][:split_idx]
        self.dates_test = dados_ml['data'][split_idx:]

        print(f"\n Dados preparados para ML:")
        print(f"   Features: {features_disponiveis}")
        print(f"   Treino: {len(X_train)} amostras")
        print(f"   Teste: {len(X_test)} amostras")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def treinar_modelo_arima(self):
        """
        Treina modelo ARIMA para séries temporais
        """

        print("\n" + "=" * 60)
        print(" Treinando Modelo ARIMA")
        print("=" * 60)

        # Usar dados diários de demanda
        series = self.dados.set_index('data')['demanda']

        # Dividir em treino e teste
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        # Auto-ARIMA para encontrar melhores parâmetros
        print("   Buscando melhores parâmetros...")

        self.modelo_arima = pm.auto_arima(
            train,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=True, m=7,  # Sazonalidade semanal
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_fits=50
        )

        print(f"\n Melhor modelo ARIMA: {self.modelo_arima.order}, {self.modelo_arima.seasonal_order}")

        # Fazer previsões
        previsoes = self.modelo_arima.predict(n_periods=len(test))

        # Calcular métricas
        mae = mean_absolute_error(test, previsoes)
        rmse = np.sqrt(mean_squared_error(test, previsoes))
        mape = np.mean(np.abs((test.values - previsoes) / test.values)) * 100

        self.metricas['ARIMA'] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'order': self.modelo_arima.order,
            'seasonal_order': self.modelo_arima.seasonal_order
        }

        print(f"\n Métricas ARIMA:")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")

        return self.modelo_arima

    def treinar_modelos_ml(self, X_train, X_test, y_train, y_test):
        """
        Treina múltiplos modelos de machine learning

        Args:
            X_train, X_test, y_train, y_test: Dados preparados
        """

        print("\n" + "=" * 60)
        print(" Treinando Modelos de Machine Learning")
        print("=" * 60)

        modelos = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        self.modelos_ml = {}

        for nome, modelo in modelos.items():
            print(f"\n Treinando {nome}...")

            # Treinar
            modelo.fit(X_train, y_train)

            # Prever
            y_pred = modelo.predict(X_test)

            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Evitar divisão por zero no MAPE
            y_test_safe = np.where(y_test == 0, 0.001, y_test)
            mape = np.mean(np.abs((y_test.values - y_pred) / y_test_safe)) * 100

            self.metricas[nome] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }

            self.modelos_ml[nome] = modelo

            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   R²: {r2:.3f}")
            print(f"   MAPE: {mape:.2f}%")

        # Identificar melhor modelo
        melhor_modelo = min(self.metricas, key=lambda x: self.metricas[x].get('mape', float('inf')))
        print(f"\n🏆 Melhor modelo: {melhor_modelo}")

        return self.modelos_ml


    #  OTIMIZAÇÃO DE ESTOQUE


    def otimizar_estoque(self, previsoes,
                         custo_armazenagem=2.0,
                         custo_falta=10.0,
                         custo_pedido=50.0,
                         lead_time=3,
                         nivel_servico=0.95):
        """
        Otimiza níveis de estoque baseado nas previsões

        Args:
            previsoes: Array com previsões de demanda
            custo_armazenagem: Custo por unidade por dia
            custo_falta: Custo por unidade em falta
            custo_pedido: Custo fixo por pedido
            lead_time: Tempo de entrega em dias
            nivel_servico: Nível de serviço desejado
        """

        print("\n" + "=" * 60)
        print(" OTIMIZAÇÃO DE ESTOQUE")
        print("=" * 60)

        # Estatísticas da previsão
        demanda_media = np.mean(previsoes)
        demanda_std = np.std(previsoes)

        # Cálculo do estoque de segurança (assumindo distribuição normal)
        z_score = {
            0.90: 1.28,
            0.95: 1.645,
            0.99: 2.326
        }.get(nivel_servico, 1.645)

        estoque_seguranca = z_score * demanda_std * np.sqrt(lead_time)

        # Ponto de pedido
        ponto_pedido = demanda_media * lead_time + estoque_seguranca

        # Lote econômico de pedido (EOQ)
        demanda_anual = demanda_media * 365
        eoq = np.sqrt((2 * demanda_anual * custo_pedido) / custo_armazenagem)

        # Nível máximo de estoque
        estoque_maximo = ponto_pedido + eoq

        # Custo total anual estimado
        custo_total = (demanda_anual / eoq) * custo_pedido + \
                      (eoq / 2) * custo_armazenagem + \
                      estoque_seguranca * custo_armazenagem

        resultados = {
            'demanda_media_diaria': demanda_media,
            'demanda_std_diaria': demanda_std,
            'estoque_seguranca': estoque_seguranca,
            'ponto_pedido': ponto_pedido,
            'lote_economico': eoq,
            'estoque_maximo': estoque_maximo,
            'custo_total_anual': custo_total,
            'lead_time': lead_time,
            'nivel_servico': nivel_servico
        }

        print("\n Parâmetros Otimizados:")
        print(f"   Demanda média diária: {demanda_media:.1f}")
        print(f"   Estoque de segurança: {estoque_seguranca:.1f} unidades")
        print(f"   Ponto de pedido: {ponto_pedido:.1f} unidades")
        print(f"   Lote econômico: {eoq:.1f} unidades")
        print(f"   Estoque máximo: {estoque_maximo:.1f} unidades")
        print(f"   Custo total anual estimado: R$ {custo_total:.2f}")

        self.parametros_estoque = resultados
        return resultados


    #  PREVISÃO FUTURA

    def prever_demanda_futura(self, dias_futuros=90, modelo='melhor'):
        """
        Faz previsões para períodos futuros

        Args:
            dias_futuros: Número de dias para prever
            modelo: 'arima', 'ml' ou 'melhor'
        """

        print(f"\n" + "=" * 60)
        print(f" PREVISÃO PARA {dias_futuros} DIAS FUTUROS")
        print("=" * 60)

        datas_futuras = pd.date_range(
            start=self.dados['data'].iloc[-1] + timedelta(days=1),
            periods=dias_futuros,
            freq='D'
        )

        if modelo == 'arima' and hasattr(self, 'modelo_arima'):
            # Previsão com ARIMA
            previsoes = self.modelo_arima.predict(n_periods=dias_futuros)

        elif modelo == 'ml' and hasattr(self, 'modelos_ml'):
            # Para ML, precisaríamos gerar features futuras
            print(" Previsão ML requer features futuras. Usando ARIMA como fallback.")
            if hasattr(self, 'modelo_arima'):
                previsoes = self.modelo_arima.predict(n_periods=dias_futuros)
            else:
                print(" Nenhum modelo ARIMA disponível.")
                return None

        else:
            # Usar melhor modelo disponível
            if hasattr(self, 'modelo_arima'):
                previsoes = self.modelo_arima.predict(n_periods=dias_futuros)
            else:
                print(" Nenhum modelo treinado disponível.")
                return None

        # Criar DataFrame de previsões
        self.previsoes_futuras = pd.DataFrame({
            'data': datas_futuras,
            'demanda_prevista': previsoes,
            'limite_inferior': previsoes * 0.8,  # Intervalo aproximado
            'limite_superior': previsoes * 1.2
        })

        # Otimizar estoque com base nas previsões
        self.otimizar_estoque(previsoes)

        # Visualizar previsões
        self._plotar_previsoes_futuras()

        return self.previsoes_futuras

    def _plotar_previsoes_futuras(self):
        """Visualiza previsões futuras"""

        fig, ax = plt.subplots(figsize=(14, 6))

        # Dados históricos recentes
        dias_historico = 180
        dados_recentes = self.dados.iloc[-dias_historico:]

        ax.plot(dados_recentes['data'], dados_recentes['demanda'],
                label='Histórico', alpha=0.7, linewidth=1)

        # Previsões futuras
        ax.plot(self.previsoes_futuras['data'], self.previsoes_futuras['demanda_prevista'],
                label='Previsão', color='red', linewidth=2)

        # Intervalo de confiança
        ax.fill_between(self.previsoes_futuras['data'],
                        self.previsoes_futuras['limite_inferior'],
                        self.previsoes_futuras['limite_superior'],
                        alpha=0.3, color='red', label='Intervalo (80%)')

        # Linha divisória
        ax.axvline(x=self.dados['data'].iloc[-1], color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f'Previsão de Demanda - {self.nome_produto}')
        ax.set_xlabel('Data')
        ax.set_ylabel('Demanda')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    #  RELATÓRIO EXECUTIVO

    def gerar_relatorio_executivo(self):
        """Gera relatório completo com todas as análises e recomendações"""

        print("\n" + "=" * 70)
        print(" RELATÓRIO EXECUTIVO - SISTEMA DE PREVISÃO DE DEMANDA")
        print("=" * 70)

        print(f"\nProduto: {self.nome_produto}")
        print(f"Data do relatório: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

        #  Resumo dos dados
        print("\n" + "-" * 50)
        print(
            f"Período analisado: {self.dados['data'].min().strftime('%d/%m/%Y')} a {self.dados['data'].max().strftime('%d/%m/%Y')}")
        print(f"Total de dias: {len(self.dados)}")
        print(f"Demanda média: {self.dados['demanda'].mean():.1f} unidades/dia")
        print(f"Demanda máxima: {self.dados['demanda'].max():.1f} unidades")
        print(f"Demanda mínima: {self.dados['demanda'].min():.1f} unidades")
        print(f"Desvio padrão: {self.dados['demanda'].std():.1f} unidades")

        # Performance dos modelos
        if self.metricas:
            for modelo, metricas in self.metricas.items():
                print(f"\n{modelo}:")
                print(f"   MAE: {metricas.get('mae', 0):.2f}")
                print(f"   RMSE: {metricas.get('rmse', 0):.2f}")
                print(f"   MAPE: {metricas.get('mape', 0):.2f}%")
                if 'r2' in metricas:
                    print(f"   R²: {metricas['r2']:.3f}")

        #  Recomendações de estoque
        if hasattr(self, 'parametros_estoque'):
            p = self.parametros_estoque
            print(f"\n Parâmetros Otimizados:")
            print(f"   • Estoque de segurança: {p['estoque_seguranca']:.0f} unidades")
            print(f"   • Ponto de pedido: {p['ponto_pedido']:.0f} unidades")
            print(f"   • Lote econômico de compra: {p['lote_economico']:.0f} unidades")
            print(f"   • Estoque máximo recomendado: {p['estoque_maximo']:.0f} unidades")
            print(f"   • Lead time considerado: {p['lead_time']} dias")
            print(f"   • Nível de serviço: {p['nivel_servico'] * 100:.0f}%")
            print(f"\n Custo total anual estimado: R$ {p['custo_total_anual']:.2f}")

        # Previsões futuras
        if hasattr(self, 'previsoes_futuras'):
            print("\n" + "-" * 50)
            print("4. PREVISÕES PARA PRÓXIMOS 90 DIAS")
            print("-" * 50)

            prox_30 = self.previsoes_futuras.head(30)
            prox_60_90 = self.previsoes_futuras.iloc[30:90]

            print(f"\nPróximos 30 dias:")
            print(f"   Demanda total prevista: {prox_30['demanda_prevista'].sum():.0f} unidades")
            print(f"   Média diária: {prox_30['demanda_prevista'].mean():.1f} unidades")
            print(f"   Pico previsto: {prox_30['demanda_prevista'].max():.0f} unidades")

            print(f"\nDias 31-90:")
            print(f"   Demanda total prevista: {prox_60_90['demanda_prevista'].sum():.0f} unidades")
            print(f"   Média diária: {prox_60_90['demanda_prevista'].mean():.1f} unidades")

        #  Ações recomendadas
        print("\n Imediatas (próximos 7 dias):")
        print("   • Revisar níveis de estoque com base no ponto de pedido")
        print("   • Preparar pedidos para períodos de alta demanda prevista")
        print("   • Ajustar políticas de desconto para produtos com estoque elevado")

        print("\n Curto Prazo (próximos 30 dias):")
        print("   • Implementar monitoramento diário das previsões")
        print("   • Ajustar parâmetros do modelo com novos dados")
        print("   • Revisar contratos com fornecedores considerando lead time")

        print("\n Médio Prazo (próximos 90 dias):")
        print("   • Avaliar necessidade de ajuste na capacidade de armazenagem")
        print("   • Desenvolver estratégias para produtos complementares")
        print("   • Implementar sistema de alerta para desvios significativos")

        print("\n" + "=" * 70)
        print("FIM DO RELATÓRIO")
        print("=" * 70)



# EXEMPLO DE USO

def main():
    """
    Função principal demonstrando o uso completo do sistema
    """

    try:
        #  Inicializar o sistema
        sistema = PrevisaoDemandaSazonal(nome_produto="Produto Sazonal Premium")

        #  Gerar dados sintéticos
        print("\n Etapa 1: Gerando dados sintéticos...")
        dados = sistema.gerar_dados_sinteticos(
            periodo_dias=1095,  # 3 anos
            tendencia_base=100,
            sazonalidade_anual=30,
            ruido=15,
            eventos_especiais=True
        )

        # Análise exploratória
        print("\n Etapa 2: Realizando análise exploratória...")
        sistema.analisar_dados()

        #  Preparar dados para ML
        print("\n Etapa 3: Preparando dados para Machine Learning...")
        X_train, X_test, y_train, y_test = sistema.preparar_dados_ml()

        #  Treinar modelos
        print("\n Etapa 4: Treinando modelos...")

        # Modelo ARIMA
        if sistema.dados is not None:
            modelo_arima = sistema.treinar_modelo_arima()

        # Modelos de ML
        if X_train is not None:
            modelos_ml = sistema.treinar_modelos_ml(X_train, X_test, y_train, y_test)

        #  Prever demanda futura
        print("\n Etapa 5: Gerando previsões futuras...")
        if hasattr(sistema, 'modelo_arima'):
            previsoes = sistema.prever_demanda_futura(dias_futuros=90)

        #  Gerar relatório executivo
        print("\n Etapa 6: Gerando relatório executivo...")
        sistema.gerar_relatorio_executivo()

        print("\n Sistema executado com sucesso!")
        return sistema

    except Exception as e:
        print(f"\n Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



# EXECUÇÃO DO SISTEMA
if __name__ == "__main__":
    sistema = main()