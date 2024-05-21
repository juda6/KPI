import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("Analyse des KPI pour une entreprise de vente en ligne")

df = pd.read_csv("atomic_data.csv")

df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
df["Revenue"] = df["Quantity"] * df["Unit Price"]

# Objectif 1 : Santé financière 
if st.sidebar.checkbox("Santé financière"):
    revenue = df["Revenue"].sum()
    st.write(f"Revenu total: {revenue}")
    cost = df["Quantity"].sum() * df["Unit Price"].mean()
    profit_margin = (revenue - cost) / revenue
    st.write(f"Marge bénéficiaire: {profit_margin:.2%}")

# Objectif 2 : Chiffre d'affaires par produit
if st.sidebar.checkbox("Chiffre d'affaires par produit"):
    product_sales = (
        df.groupby("Product Name")["Revenue"].sum().sort_values(ascending=False)
    )
    st.write(f"Chiffre d'affaires par produit :", product_sales)
    
# Objectif 3 : Moyen de paiement le plus utilisé
if st.sidebar.checkbox("Moyen de paiement le plus utilisé"):
    payment_methods = df["Payment Method"].value_counts()
    st.write(f"Moyen de paiement le plus utilisé :", payment_methods)

# Objectif 4 : Ventes par pays
if st.sidebar.checkbox("Ventes par pays"):
    country_sales = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    st.write(f"Ventes par pays :", country_sales)

# Objectif 5 : Tendance des ventes en fonction du temps
if st.sidebar.checkbox("Tendance des ventes"):
    sales_trend = df.groupby(df['Transaction Date'])['Revenue'].sum()
    st.write(f"Tendance des ventes :", sales_trend)
    st.write("graphique linéaire de tendance des ventes")
    st.line_chart(sales_trend)

# Objectif 6 : Prédiction du chiffre d'affaires pour Mai 2024
if st.sidebar.checkbox("Prédiction du chiffre d'affaires pour Mai 2024"):
    # Préparer les données pour Prophet
    df_prophet = df[['Transaction Date', 'Revenue']].rename(columns={'Transaction Date': 'ds', 'Revenue': 'y'})

    # Initialiser et entraîner le modèle Prophet
    model = Prophet()
    model.fit(df_prophet)

    #  prévisions pour les 12 prochains mois
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Filtrage des prévisions pour le mois de Mai 2024
    forecast_may_2024 = forecast[(forecast['ds'] >= '2024-05-01') & (forecast['ds'] < '2024-06-01')]
    total_revenue_may_2024 = forecast_may_2024['yhat'].sum()

    # Faire afficher les prédictions
    st.subheader(f"Prédiction du chiffre d'affaires pour Mai 2024 : {total_revenue_may_2024:.2f}")
