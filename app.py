def relatorio_previsao(self, cbo_codigo, anos_futuros=[5,10,15,20]):
    df = self.df
    col_cbo = "cbo2002ocupação"
    col_data = "competênciamov"
    col_salario = "salário"
    saldo_col = "saldomovimentação"

    prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
    st.subheader(f"Profissão: {prof_info.iloc[0]['cbo_descricao']}" if not prof_info.empty else f"CBO: {cbo_codigo}")
    df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
    if df_cbo.empty:
        st.warning("Nenhum registro encontrado para a profissão selecionada.")
        return

    st.write(f"**Registros encontrados:** {len(df_cbo):,}")
    with st.expander("Perfil Demográfico"):
        if 'idade' in df_cbo.columns:
            idade_media = pd.to_numeric(df_cbo['idade'], errors='coerce').mean()
            st.write(f"Idade média: {idade_media:.1f} anos")
        if 'sexo' in df_cbo.columns:
            sexo_map = {'1.0':'Masculino','3.0':'Feminino','1':'Masculino','3':'Feminino'}
            masculino = df_cbo['sexo'].apply(lambda x: sexo_map.get(str(x), str(x))).value_counts().get('Masculino', 0)
            feminino  = df_cbo['sexo'].apply(lambda x: sexo_map.get(str(x), str(x))).value_counts().get('Feminino', 0)
            st.write(f"Homens: {masculino} | Mulheres: {feminino}")
        if 'graudeinstrucao' in df_cbo.columns:
            escolaridade = df_cbo['graudeinstrucao'].value_counts().head(3)
            escolaridade_map = {
                '1': 'Analfabeto','2': 'Até 5ª inc. Fundamental','3': '5ª completo Fundamental',
                '4': '6ª a 9ª Fundamental','5': 'Fundamental completo','6': 'Médio incompleto',
                '7': 'Médio completo','8': 'Superior incompleto','9': 'Superior completo',
                '10': 'Mestrado','11': 'Doutorado','80':'Pós-graduação'
            }
            esc_strings = []
            for nivel,count in escolaridade.items():
                nivel_nome = escolaridade_map.get(str(int(float(nivel))), str(nivel))
                esc_strings.append(f"{nivel_nome}: {(count/len(df_cbo))*100:.1f}%")
            st.write("Principais níveis:", ", ".join(esc_strings))
        if 'uf' in df_cbo.columns:
            uf_map = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
            uf_dist = df_cbo['uf'].value_counts().head(5)
            uf_lista = [f"{uf_map.get(str(int(float(uf))),str(uf))}: {count:,} ({(count/len(df_cbo))*100:.1f}%)"
                        for uf,count in uf_dist.items()]
            st.write("Principais UF:", ", ".join(uf_lista))

    # --- Situação do Mercado de Trabalho ---
    st.subheader("Situação do Mercado de Trabalho")
    saldo_col = "saldomovimentação"
    if saldo_col in df_cbo.columns:
        df_cbo[saldo_col] = pd.to_numeric(df_cbo[saldo_col], errors='coerce')
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors='coerce')
        df_cbo['ano'] = df_cbo[col_data].dt.year
        saldo_ano = df_cbo.groupby("ano")[saldo_col].sum().reset_index()
        st.markdown("**Histórico do saldo anual:**")
        linhas_historico = []
        for _, linha in saldo_ano.iterrows():
            v = int(linha[saldo_col])
            if v > 0: status = "Expansão"
            elif v < 0: status = "Retração"
            else: status = "Estável"
            linhas_historico.append(f"Ano {int(linha['ano'])}: {v:+,} ({status})")
        st.write("\n".join(linhas_historico))
        # Previsão individual para os anos futuros
        X_hist = saldo_ano[['ano']]
        y_hist = saldo_ano[saldo_col]
        if len(X_hist) > 1:
            model = LinearRegression().fit(X_hist, y_hist)
            ano_max = int(saldo_ano['ano'].max())
            linhas_prev = []
            for a in anos_futuros:
                ano_futuro = ano_max + a
                pred = int(model.predict(np.array([[ano_futuro]]))[0])
                if pred > 0: status = "Expansão"
                elif pred < 0: status = "Retração"
                else: status = "Estável"
                linhas_prev.append(f"Previsão para {ano_futuro}: {pred:+,} ({status})")
            st.markdown("**Previsão do saldo futuro:**")
            st.write("\n".join(linhas_prev))
    else:
        st.write("Sem dados de saldo de movimentação para esta profissão.")

    # --- PREVISÃO SALARIAL ---
    st.subheader("Previsão Salarial (próximos anos)")
    df_cbo[col_salario] = pd.to_numeric(df_cbo[col_salario].astype(str).str.replace(",",".").str.replace(" ",""), errors="coerce")
    df_cbo = df_cbo.dropna(subset=[col_salario])
    df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors='coerce')
    df_cbo = df_cbo.dropna(subset=[col_data])
    if df_cbo.empty:
        st.warning("Não há dados temporais válidos.")
        return
    df_cbo['tempo_meses'] = ((df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month)
    df_mensal = df_cbo.groupby('tempo_meses')[col_salario].mean().reset_index()
    salario_atual = df_cbo[col_salario].mean()
    st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")
    if len(df_mensal) >= 2:
        X_m = df_mensal[['tempo_meses']]
        y_m = df_mensal[col_salario]
        model_sal = LinearRegression().fit(X_m, y_m)
        ult_mes = int(df_mensal['tempo_meses'].max())
        resultados = []
        for anos in anos_futuros:
            mes_futuro = ult_mes + anos * 12
            ano_futuro = 2020 + mes_futuro // 12
            pred = model_sal.predict(np.array([[mes_futuro]]))[0]
            variacao = ((pred-salario_atual)/salario_atual)*100
            resultados.append(f"Previsão para {ano_futuro}: R$ {self.formatar_moeda(pred)}  ({variacao:+.1f}%)")
        st.write("\n".join(resultados))
    else:
        st.info("Previsão baseada apenas na média atual.")
