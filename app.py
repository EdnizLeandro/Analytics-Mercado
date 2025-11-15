import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class MercadoTrabalhoPredictor:
    def __init__(self, csv_files: list, codigos_filepath: str):
        self.csv_files = csv_files
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None

    def formatar_moeda(self, valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def carregar_dados(self):
        print("Etapa 1: Carregando datasets .csv...")
        dfs = [pd.read_csv(path, encoding='utf-8', sep=';', on_bad_lines='skip') for path in self.csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Dataset carregado com {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.")
        print("Carregando lista de códigos CBO...")
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)
        print(f"{len(self.df_codigos)} profissões carregadas.\n")

    def limpar_dados(self):
        print("Etapa 2: Limpando dataset...")
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True
        print(f"Limpeza finalizada! Dataset agora tem {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.\n")
        print("Identificando colunas automaticamente...")
        self._identificar_colunas()

    def _identificar_colunas(self):
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocup' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower and 'mov' in col_lower:
                self.coluna_data = col
            if 'salário' in col_lower:
                self.coluna_salario = col
        print(f"  ✓ Coluna CBO: {self.coluna_cbo}")
        print(f"  ✓ Coluna DATA: {self.coluna_data}")
        print(f"  ✓ Coluna SALÁRIO: {self.coluna_salario}\n")

    def buscar_profissao(self, entrada: str) -> pd.DataFrame:
        if not self.cleaned:
            print("Limpe o dataset antes de buscar profissões.")
            return pd.DataFrame()
        print(f"Etapa 3: Buscando '{entrada}'...")
        if entrada.isdigit():
            resultados = self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]
            if not resultados.empty:
                print(f"CBO {entrada} encontrado: {resultados.iloc[0]['cbo_descricao']}\n")
                return resultados
            else:
                print("Código CBO não encontrado.\n")
                return pd.DataFrame()
        mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
        resultados = self.df_codigos[mask]
        if resultados.empty:
            print("Nenhuma profissão encontrada com esse nome.\n")
        else:
            print(f"{len(resultados)} profissão(ões) encontrada(s):\n")
            for idx, row in resultados.iterrows():
                print(f"  [{row['cbo_codigo']}] {row['cbo_descricao']}")
            print()
        return resultados

    def prever_mercado(self, cbo_codigo: str, anos_futuros=[5, 10, 15, 20]):
        if not self.cleaned:
            print("Dataset não limpo.")
            return
        if not all([self.coluna_cbo, self.coluna_data, self.coluna_salario]):
            print("Colunas não identificadas.")
            return
        print(f"Etapa 4: Prevendo mercado para CBO {cbo_codigo}...")
        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        if not prof_info.empty:
            print(f"Profissão: {prof_info.iloc[0]['cbo_descricao']}\n")
        print(f"Filtrando dados para CBO {cbo_codigo}...")
        df_cbo = self.df[self.df[self.coluna_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            print("Nenhum registro encontrado.\n")
            return
        print(f"{len(df_cbo):,} registros encontrados.\n")
        print("="*70)
        print("ANÁLISE DO MERCADO DE TRABALHO")
        print("="*70)
        if 'saldomovimentacao' in df_cbo.columns:
            saldo_total = df_cbo['saldomovimentacao'].sum()
            print(f"\nSALDO DE MOVIMENTAÇÃO (Admissões - Desligamentos):")
            print(f"   • Saldo total no período: {saldo_total:+,.0f} postos de trabalho")
            if saldo_total > 0:
                print(f"   Mercado em EXPANSÃO (mais admissões que desligamentos)")
            elif saldo_total < 0:
                print(f"   Mercado em RETRAÇÃO (mais desligamentos que admissões)")
            else:
                print(f"   ➡️ Mercado ESTÁVEL")

        print(f"\nPERFIL DEMOGRÁFICO:")
        if 'idade' in df_cbo.columns:
            idade_media = df_cbo['idade'].mean()
            print(f"   • Idade média: {idade_media:.1f} anos")
        if 'sexo' in df_cbo.columns:
            sexo_dist = df_cbo['sexo'].value_counts()
            print(f"   • Distribuição por sexo:")
            for sexo, count in sexo_dist.items():
                pct = (count / len(df_cbo)) * 100
                sexo_label = "Masculino" if str(sexo) == "1.0" else "Feminino" if str(sexo) == "3.0" else sexo
                print(f"      - {sexo_label}: {pct:.1f}%")
        if 'graudeinstrucao' in df_cbo.columns:
            escolaridade = df_cbo['graudeinstrucao'].value_counts().head(3)
            print(f"   • Principais níveis de escolaridade:")
            escolaridade_map = {
                '1': 'Analfabeto',
                '2': 'Até 5ª inc. Fundamental',
                '3': '5ª completo Fundamental',
                '4': '6ª a 9ª Fundamental',
                '5': 'Fundamental completo',
                '6': 'Médio incompleto',
                '7': 'Médio completo',
                '8': 'Superior incompleto',
                '9': 'Superior completo',
                '10': 'Mestrado',
                '11': 'Doutorado',
                '80': 'Pós-graduação'
            }
            for nivel, count in escolaridade.items():
                pct = (count / len(df_cbo)) * 100
                try:
                    nivel_nome = escolaridade_map.get(str(int(float(nivel))), str(nivel))
                except:
                    nivel_nome = str(nivel)
                print(f"      - {nivel_nome}: {pct:.1f}%")
        if 'uf' in df_cbo.columns:
            print(f"\nDISTRIBUIÇÃO GEOGRÁFICA:")
            uf_map = {
                '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
                '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL',
                '28': 'SE', '29': 'BA', '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP', '41': 'PR',
                '42': 'SC', '43': 'RS', '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
            }
            uf_dist = df_cbo['uf'].value_counts().head(5)
            for uf_cod, count in uf_dist.items():
                pct = (count / len(df_cbo)) * 100
                try:
                    uf_nome = uf_map.get(str(int(float(uf_cod))), str(uf_cod))
                except:
                    uf_nome = str(uf_cod)
                print(f"   • {uf_nome}: {count:,} registros ({pct:.1f}%)")
        if 'tipoempregador' in df_cbo.columns:
            print(f"\nTIPO DE EMPREGADOR:")
            tipo_map = {'0': 'CNPJ', '1': 'CPF', '2': 'CNO (Obra)', '3': 'CAEPF'}
            tipo_emp = df_cbo['tipoempregador'].value_counts().head(3)
            for tipo, count in tipo_emp.items():
                pct = (count / len(df_cbo)) * 100
                try:
                    tipo_nome = tipo_map.get(str(int(float(tipo))), str(tipo))
                except:
                    tipo_nome = str(tipo)
                print(f"   • {tipo_nome}: {pct:.1f}%")

        print(f"\n{'='*70}")
        print("PREVISÃO SALARIAL")
        print("="*70)

        # Conversão robusta do salário (trata ',' de decimal, espaços e valores inválidos)
        df_cbo[self.coluna_salario] = pd.to_numeric(
            df_cbo[self.coluna_salario].astype(str).str.replace(",", ".").str.replace(" ", ""),
            errors='coerce'
        )
        df_cbo = df_cbo.dropna(subset=[self.coluna_salario])

        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        if df_cbo.empty:
            print("Não há dados temporais válidos.\n")
            return
        df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                  df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()
        salario_atual = df_cbo[self.coluna_salario].mean()
        print(f"\nSalário médio atual: R$ {self.formatar_moeda(salario_atual)}")
        if len(df_mensal) < 2:
            print(f"Previsão baseada na média atual:")
            for anos in anos_futuros:
                print(f"  {anos} anos → R$ {self.formatar_moeda(salario_atual)}")
        else:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[self.coluna_salario]
            model = LinearRegression()
            model.fit(X, y)
            print(f"Período analisado: {df_cbo[self.coluna_data].min().strftime('%m/%Y')} a {df_cbo[self.coluna_data].max().strftime('%m/%Y')}")
            ult_mes = df_mensal['tempo_meses'].max()
            print(f"\nPrevisões de salário médio:")
            for anos in anos_futuros:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                variacao = ((pred - salario_atual) / salario_atual) * 100
                print(f"  {anos} anos → R$ {self.formatar_moeda(max(pred, 0))} ({variacao:+.1f}%)")

        if 'saldomovimentacao' in df_cbo.columns:
            print(f"\n{'='*70}")
            print("PREVISÃO DE TENDÊNCIA DO MERCADO")
            print("="*70)
            df_saldo_mensal = df_cbo.groupby('tempo_meses')['saldomovimentacao'].sum().reset_index()
            if len(df_saldo_mensal) >= 2:
                X_saldo = df_saldo_mensal[['tempo_meses']]
                y_saldo = df_saldo_mensal['saldomovimentacao']
                model_saldo = LinearRegression()
                model_saldo.fit(X_saldo, y_saldo)
                print(f"\nTendência de saldo de vagas:")
                ult_mes = df_saldo_mensal['tempo_meses'].max()
                for anos in anos_futuros:
                    mes_futuro = ult_mes + anos * 12
                    pred_saldo = model_saldo.predict(np.array([[mes_futuro]]))[0]
                    if pred_saldo > 100:
                        tendencia, descricao = "ALTA DEMANDA", "Forte crescimento esperado"
                    elif pred_saldo > 50:
                        tendencia, descricao = "CRESCIMENTO MODERADO", "Expansão gradual"
                    elif pred_saldo > 0:
                        tendencia, descricao = "CRESCIMENTO LEVE", "Pequena expansão"
                    elif pred_saldo > -50:
                        tendencia, descricao = "RETRAÇÃO LEVE", "Pequena diminuição"
                    elif pred_saldo > -100:
                        tendencia, descricao = "RETRAÇÃO MODERADA", "Redução gradual"
                    else:
                        tendencia, descricao = "RETRAÇÃO FORTE", "Forte diminuição"
                    print(f"  {anos} anos → {pred_saldo:+,.0f} vagas/mês - {tendencia}")
                    print(f"           {descricao}")
            else:
                saldo_total = df_cbo['saldomovimentacao'].sum()
                if saldo_total > 100:
                    base_tendencia, base_descricao = "ALTA DEMANDA", "Mercado em expansão"
                elif saldo_total > 0:
                    base_tendencia, base_descricao = "CRESCIMENTO LEVE", "Mercado em leve crescimento"
                elif saldo_total > -100:
                    base_tendencia, base_descricao = "RETRAÇÃO LEVE", "Mercado em leve retração"
                else:
                    base_tendencia, base_descricao = "RETRAÇÃO FORTE", "Mercado em retração"
                print(f"   Status atual: {base_tendencia} ({base_descricao})")
                print(f"\nProjeção (mantendo tendência atual):")
                for anos in anos_futuros:
                    print(f"  {anos} anos → {base_tendencia}")
        print("\n" + "="*70 + "\n")

def main():
    csv_files = [
        "2020_PE1.csv",
        "2021_PE1.csv",
        "2022_PE1.csv",
        "2023_PE1.csv",
        "2024_PE1.csv",
        "2025_PE1.csv"
    ]
    codigos_filepath = "cbo.xlsx"
    app = MercadoTrabalhoPredictor(csv_files, codigos_filepath)
    app.carregar_dados()
    app.limpar_dados()
    while True:
        entrada = input("Digite o nome ou código da profissão (ou 'sair' para encerrar): ").strip()
        if entrada.lower() == 'sair':
            print("Encerrando aplicação.")
            break
        resultados = app.buscar_profissao(entrada)
        if resultados.empty:
            continue
        if len(resultados) == 1:
            cbo = resultados['cbo_codigo'].iloc[0]
            print(f"Código CBO selecionado: {cbo}\n")
        else:
            cbo = input("\nDigite o código CBO desejado: ").strip()
        app.prever_mercado(cbo)

if __name__ == "__main__":
    main()
