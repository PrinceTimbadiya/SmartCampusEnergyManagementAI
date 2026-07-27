"""
About Page
Smart Campus Energy Management Dashboard
Student: Prince Timbadiya
"""

import streamlit as st

st.markdown(
    """
<div class="section-header">ℹ️ About This Project</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="chart-container">
    <h3>📌 Project Overview</h3>
    <p>
        The <strong>Smart Campus Energy Management System</strong> is an AI-enabled solution
        designed to optimize energy consumption across university campuses. By leveraging
        machine learning, the system predicts energy usage patterns and provides actionable
        recommendations for reducing waste and improving sustainability.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    <div class="chart-container">
        <h3>🎯 Problem Statement</h3>
        <ul>
            <li>Academic Block usage surged from 145 to 182 kWh (+25%)</li>
            <li>Hostel peaked at 210-215 kWh on 16-17 June</li>
            <li>Consumption tracks temperature (34°C → 41°C)</li>
            <li>Lights/fans left ON in empty classrooms</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="chart-container">
        <h3>🚀 Solution</h3>
        <ul>
            <li>AI-powered energy prediction</li>
            <li>Real-time monitoring dashboard</li>
            <li>Proactive alert system</li>
            <li>Data-driven recommendations</li>
            <li>Building-specific optimization</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown(
    """
<div class="chart-container">
<h3>🛠️ Technology Stack</h3>

<table width="100%">
<tr>
<td align="center">🐍<br><b>Python</b></td>
<td align="center">🤖<br><b>Scikit-learn</b></td>
<td align="center">📊<br><b>Pandas</b></td>
</tr>

<tr>
<td align="center">📈<br><b>Plotly</b></td>
<td align="center">🌊<br><b>Streamlit</b></td>
<td align="center">🔢<br><b>NumPy</b></td>
</tr>

</table>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
<div class="chart-container">
    <h3>👨‍🎓 Student Information</h3>
    <p>
        <strong>Name:</strong> Prince Timbadiya<br>
        <strong>Course:</strong> Bachelor of Computer Applications (BCA)<br>
        <strong>Internship:</strong> Microsoft + 1M1B Green Skills & Applied AI Internship<br>
        <strong>Year:</strong> 2026
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
<div class="chart-container">
    <h3>🔮 Future Scope</h3>
    <ul>
        <li>IoT sensor integration for real-time data</li>
        <li>Solar generation prediction for hybrid optimization</li>
        <li>Mobile app for campus-wide access</li>
        <li>Gamification for student engagement</li>
        <li>Water and waste management integration</li>
        <li>Carbon footprint tracking</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
<div class="footer">
    <strong>Smart Campus Energy Management</strong> · Developed by <strong>Prince Timbadiya</strong> · 
    Microsoft + 1M1B Green Skills & Applied AI Internship · 2026
</div>
""",
    unsafe_allow_html=True,
)
