import streamlit as st


def show_about():
    st.header("Quiénes Somos")

    st.markdown(
        """
        Esta aplicación ha sido creada por miembros del **Grupo de Fisiopatología del Calcio Intracelular**
        del **IBGM (CSIC-UVa)**.
        """
    )

    st.subheader("Equipo")
    st.markdown(
        """
        - **Dra. María Elena Hernando Pérez**  
          Correo: **mariaelena.hernando@uva.es**
        - **Dr. Enrique Pérez Riesgo**  
          Correo: **epercamh@gmail.com**
        """
    )

    st.subheader("Dirección del grupo")
    st.markdown(
        """
        El grupo está dirigido por:

        - **Dr. Carlos Villalobos Jorge**
        - **Dra. Lucía Núñez Llorente**
        """
    )