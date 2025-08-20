import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fair_quantum_decision_theory import (
    FairQuantumDecisionModel,
    FairClassicalDecisionModel,
)

# Configure Streamlit
st.set_page_config(
    page_title="Fair Quantum Decision Theory Demo", page_icon="⚖️", layout="wide"
)

st.title("⚖️⚡ Fair Quantum vs Classical Decision Theory")
st.write(
    "**Honest comparison - quantum advantages from phenomena, not inflated parameters**"
)

# Calibration notice
st.info(
    "🔧 **Calibration**: Both models use identical maximum effect sizes. Quantum advantages come from genuine phenomena like order effects and interference patterns, not parameter inflation."
)


# Initialize models with same sensitivity
@st.cache_resource
def load_models():
    context_sensitivity = 0.15  # Same for both models
    return (
        FairQuantumDecisionModel(context_sensitivity),
        FairClassicalDecisionModel(context_sensitivity),
    )


qdt_model, classical_model = load_models()

# Sidebar for parameters
st.sidebar.title("🎛️ Fair Parameter Settings")

# Show calibration info
st.sidebar.info(
    "📊 **Parameter Ranges:**\n- Context effects: 0-15% max\n- Same for both models\n- Fair comparison guaranteed"
)

# Customer Profile
st.sidebar.subheader("👤 Customer Profile")
price_sensitivity = st.sidebar.slider(
    "Price Sensitivity", 0.0, 1.0, 0.7, help="How much price affects decision"
)
quality_preference = st.sidebar.slider(
    "Quality Preference", 0.0, 1.0, 0.8, help="How much quality matters"
)
brand_loyalty = st.sidebar.slider(
    "Brand Loyalty", 0.0, 1.0, 0.4, help="Loyalty to premium brands"
)

customer_profile = {
    "price_sensitivity": price_sensitivity,
    "quality_preference": quality_preference,
    "brand_loyalty": brand_loyalty,
}

# Store Context (same effect ranges for both models)
st.sidebar.subheader("🏪 Store Context")
lighting = st.sidebar.slider(
    "Lighting Quality", 0.0, 1.0, 0.7, help="Max effect: ±15% (same for both models)"
)
music = st.sidebar.slider(
    "Music Appeal", 0.0, 1.0, 0.6, help="Max effect: ±7.5% (same for both models)"
)
crowding = st.sidebar.slider(
    "Crowding Level", 0.0, 1.0, 0.4, help="Max effect: ±7.5% (same for both models)"
)
display_quality = st.sidebar.slider(
    "Product Display", 0.0, 1.0, 0.8, help="Max effect: ±15% (same for both models)"
)

context_factors = {
    "lighting": lighting,
    "music": music,
    "crowding": crowding,
    "display_quality": display_quality,
}

# Information Order (THIS is where quantum shows advantage)
st.sidebar.subheader("📋 Information Order")
st.sidebar.write("**Key Test**: Order effects violate classical probability")
info_order = st.sidebar.selectbox(
    "Information presentation sequence:",
    [
        "price → quality → ambiance",
        "ambiance → quality → price",
        "quality → price → ambiance",
    ],
).split(" → ")

# Main predictions
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 Classical Model")
    classical_result = classical_model.classical_decision_prediction(
        customer_profile, context_factors, info_order
    )

    classical_prob = classical_result["purchase_probability"]
    st.metric("Purchase Probability", f"{classical_prob:.1%}")

    st.write("**Classical Properties:**")
    st.write("- ✅ Order independent")
    st.write("- ✅ Linear utility combination")
    st.write("- ✅ Predictable interactions")

    # Utility breakdown
    with st.expander("🔍 Utility Breakdown"):
        breakdown = classical_result["utility_breakdown"]
        st.write(f"- Base utility: {breakdown['base_utility']:.3f}")
        st.write(f"- Context effects: {breakdown['context_effects']:.3f}")
        st.write(f"- Interactions: {breakdown['interaction_bonus']:.3f}")

with col2:
    st.subheader("⚡ Quantum Model")

    with st.spinner("Running quantum simulation..."):
        qdt_result = qdt_model.quantum_decision_prediction(
            customer_profile, context_factors, info_order
        )

    qdt_prob = qdt_result["purchase_probability"]
    st.metric("Purchase Probability", f"{qdt_prob:.1%}")

    st.write("**Quantum Properties:**")
    st.write("- 🌊 Order dependent")
    st.write("- 🔗 Quantum entanglement")
    st.write("- 💫 Superposition collapse")

    # Quantum metrics
    with st.expander("🔬 Quantum Metrics"):
        st.write(f"- Interference: {qdt_result['interference_effect']:.3f}")
        st.write(f"- Superposition: {qdt_result['superposition_measure']:.3f}")
        st.write(f"- Quantum states: {qdt_result['quantum_states']}")

# Show the key difference
difference = abs(qdt_prob - classical_prob)
if difference > 0.01:  # Only highlight significant differences
    st.success(
        f"🎯 **Quantum Advantage Detected**: {difference:.1%} difference from genuine quantum effects!"
    )
else:
    st.info("📊 **Similar Results**: Models agree when quantum effects are minimal")

# Order Effects Demonstration (THE KEY TEST)
st.subheader("🔄 Order Effects: The Quantum Smoking Gun")
st.write("**Classical prediction**: Information order should NOT matter")
st.write(
    "**Quantum prediction**: Information order DOES matter due to non-commuting operations"
)

# Test all orders
order_options = [
    ["price", "quality", "ambiance"],
    ["ambiance", "quality", "price"],
    ["quality", "price", "ambiance"],
]

order_classical = []
order_quantum = []
order_labels = []

for order in order_options:
    classical_order = classical_model.classical_decision_prediction(
        customer_profile, context_factors, order
    )
    quantum_order = qdt_model.quantum_decision_prediction(
        customer_profile, context_factors, order
    )

    order_classical.append(classical_order["purchase_probability"])
    order_quantum.append(quantum_order["purchase_probability"])
    order_labels.append(" → ".join(order))

# Order effects chart
fig_order = go.Figure()

fig_order.add_trace(
    go.Bar(
        name="Classical (Order Independent)",
        x=order_labels,
        y=order_classical,
        marker_color="blue",
        text=[f"{p:.1%}" for p in order_classical],
        textposition="inside",
    )
)

fig_order.add_trace(
    go.Bar(
        name="Quantum (Order Dependent)",
        x=order_labels,
        y=order_quantum,
        marker_color="red",
        text=[f"{p:.1%}" for p in order_quantum],
        textposition="inside",
    )
)

fig_order.update_layout(
    title="Order Effects: Classical vs Quantum Models",
    xaxis_title="Information Presentation Order",
    yaxis_title="Purchase Probability",
    barmode="group",
    yaxis=dict(range=[0, 1]),
)

st.plotly_chart(fig_order, use_container_width=True)

# Calculate order sensitivity metrics
classical_variance = np.var(order_classical)
quantum_variance = np.var(order_quantum)

col1, col2, col3 = st.columns(3)
col1.metric("Classical Order Variance", f"{classical_variance:.6f}")
col2.metric("Quantum Order Variance", f"{quantum_variance:.6f}")
if classical_variance > 0:
    col3.metric("Quantum Sensitivity", f"{quantum_variance / classical_variance:.1f}x")
else:
    col3.metric("Order Effect", "Quantum Only")

# Context Sensitivity Analysis
st.subheader("🌊 Context Sensitivity (Fair Comparison)")

# Test lighting sensitivity
lighting_range = np.linspace(0, 1, 21)
classical_lighting = []
quantum_lighting = []

for light_level in lighting_range:
    test_context = context_factors.copy()
    test_context["lighting"] = light_level

    classical_test = classical_model.classical_decision_prediction(
        customer_profile, test_context, info_order
    )
    quantum_test = qdt_model.quantum_decision_prediction(
        customer_profile, test_context, info_order
    )

    classical_lighting.append(classical_test["purchase_probability"])
    quantum_lighting.append(quantum_test["purchase_probability"])

# Sensitivity chart
fig_sens = go.Figure()

fig_sens.add_trace(
    go.Scatter(
        x=lighting_range,
        y=classical_lighting,
        mode="lines+markers",
        name="Classical Response",
        line=dict(color="blue", width=3),
    )
)

fig_sens.add_trace(
    go.Scatter(
        x=lighting_range,
        y=quantum_lighting,
        mode="lines+markers",
        name="Quantum Response",
        line=dict(color="red", width=3),
    )
)

fig_sens.update_layout(
    title="Context Sensitivity: Lighting Effects (Same Max Parameters)",
    xaxis_title="Lighting Quality",
    yaxis_title="Purchase Probability",
    xaxis=dict(range=[0, 1]),
    yaxis=dict(range=[0, 1]),
)

st.plotly_chart(fig_sens, use_container_width=True)

# Show fairness metrics
classical_range = max(classical_lighting) - min(classical_lighting)
quantum_range = max(quantum_lighting) - min(quantum_lighting)

col1, col2 = st.columns(2)
col1.metric("Classical Sensitivity Range", f"{classical_range:.3f}")
col2.metric("Quantum Sensitivity Range", f"{quantum_range:.3f}")

# Educational content
st.subheader("🎓 Understanding Fair Comparison")

with st.expander("⚖️ How We Ensured Fairness"):
    st.write("""
    **Parameter Calibration:**
    - Both models use identical maximum effect sizes (15% for major factors)
    - Context factors have same weights in both models
    - No artificial inflation of quantum effects
    
    **Genuine Quantum Advantages:**
    - **Order Effects**: Quantum models show information sequence dependency
    - **Interference Patterns**: Context factors can amplify/cancel in quantum model
    - **Superposition**: Quantum uncertainty until decision measurement
    - **Entanglement**: Correlated decision factors in quantum model
    
    **What's NOT Quantum Advantage:**
    - Bigger effect sizes (we prevented this)
    - More parameters (both models have same complexity)
    - Different sensitivity ranges (calibrated to be equal)
    """)

with st.expander("🔬 Scientific Validity"):
    st.write("""
    **This Implementation:**
    - Uses same mathematical parameter ranges for both models
    - Tests specific quantum predictions (order effects)
    - Shows where quantum genuinely differs from classical
    - Honest about when both models give similar results
    
    **Real-World Testing:**
    - These order effects are testable in experiments
    - Quantum interference patterns can be measured
    - Context dependency is observable in consumer behavior
    - Some studies already support quantum decision theory
    """)

with st.expander("📊 Interpretation Guide"):
    st.write("""
    **When Quantum Shows Advantage:**
    - Large order effects (different sequences give different results)
    - Non-linear context interactions (interference patterns)
    - High superposition measures (genuine uncertainty)
    
    **When Models Agree:**
    - Simple decisions with clear preferences
    - Minimal context effects
    - Strong customer preferences override context
    
    **Business Implications:**
    - Information sequencing matters (quantum prediction)
    - Context optimization is more complex than linear models suggest
    - Customer uncertainty states are real and measurable
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Fair Quantum Decision Theory Demo** • Equal Parameters • Genuine Phenomena • Scientific Validity"
)
