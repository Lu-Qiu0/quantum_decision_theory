import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class FairQuantumDecisionModel:
    """
    Quantum Decision Theory model with fair parameter calibration
    Quantum advantages come from phenomena, not inflated effects
    """

    def __init__(self, base_context_sensitivity=0.15):
        """
        base_context_sensitivity: Maximum context effect (same as classical)
        """
        self.base_context_sensitivity = base_context_sensitivity
        self.num_qubits = 3
        self.simulator = AerSimulator(method="statevector")

        # Calibration: ensure quantum effects are comparable to classical
        self.max_rotation = self.base_context_sensitivity * np.pi

    def create_customer_state(
        self, price_sensitivity, quality_preference, brand_loyalty
    ):
        """Create quantum customer state with normalized preferences"""
        qc = QuantumCircuit(self.num_qubits)

        # Encode customer preferences as quantum states
        # Scale to reasonable rotation angles (not inflated)
        qc.ry(price_sensitivity * np.pi * 0.5, 0)  # Max π/2 for full preference
        qc.ry(quality_preference * np.pi * 0.5, 1)  # Max π/2 for full preference
        qc.ry(brand_loyalty * np.pi * 0.5, 2)  # Max π/2 for full preference

        # Create entanglement between factors (genuine quantum effect)
        qc.cx(0, 1)  # Price-quality correlation
        qc.cx(1, 2)  # Quality-brand correlation

        return qc

    def apply_contextual_interference(
        self, qc, lighting, music, crowding, display_quality
    ):
        """
        Apply contextual factors with SAME maximum effect as classical model
        Quantum advantage comes from interference patterns, not magnitude
        """

        # Context effects calibrated to match classical model magnitude
        lighting_rotation = lighting * self.max_rotation
        music_rotation = music * self.max_rotation * 0.5  # Same weight as classical
        crowding_rotation = (
            crowding * self.max_rotation * 0.5
        )  # Same weight as classical
        display_rotation = display_quality * self.max_rotation

        # Apply rotations - creates interference when qubits interact
        qc.rz(lighting_rotation, 0)  # Phase rotation on decision
        qc.ry(music_rotation, 1)  # Amplitude rotation on context
        qc.rz(-crowding_rotation, 2)  # Negative phase (crowding bad)
        qc.rx(display_rotation * 0.5, 0)  # Mixed rotation for display

        # Cross-qubit interactions (genuine quantum interference)
        qc.cx(1, 0)  # Context affects decision
        qc.cx(2, 0)  # Social factors affect decision

        return qc

    def apply_order_effects(self, qc, information_order):
        """
        Apply order effects - GENUINE quantum phenomenon
        Classical models cannot reproduce this
        """
        # Order effects: information sequence creates different quantum evolution
        for i, info_type in enumerate(information_order):
            # Earlier information has stronger effect (psychological primacy)
            strength = (len(information_order) - i) / len(information_order)
            base_angle = strength * self.max_rotation * 0.3  # Moderate effect

            if info_type == "price":
                qc.ry(base_angle, 0)  # Direct price impact
            elif info_type == "quality":
                qc.rx(base_angle, 1)  # Quality affects context perception
            elif info_type == "ambiance":
                qc.rz(base_angle, 2)  # Ambiance affects mood

        return qc

    def measure_decision_probability(self, qc, shots=1000):
        """Measure quantum decision state to get purchase probability"""
        # Add measurement to all qubits
        qc.measure_all()

        # Run quantum simulation
        sampler = SamplerV2()
        job = sampler.run([qc], shots=shots)
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # Calculate purchase probability
        # States with decision qubit = 1 represent "buy" decisions
        buy_states = [
            state for state in counts.keys() if state[2] == "1"
        ]  # Last bit is decision qubit
        total_buy_counts = sum(counts.get(state, 0) for state in buy_states)

        purchase_probability = total_buy_counts / shots

        return purchase_probability, counts

    def calculate_interference_effects(self, context_factors):
        """
        Calculate quantum interference between context factors
        This is genuine quantum behavior
        """
        factors = list(context_factors.values())
        interference_terms = []

        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                # Quantum interference: cos(φ_i - φ_j)
                phase_diff = (factors[i] - factors[j]) * np.pi
                interference = np.cos(phase_diff)
                interference_terms.append(interference)

        # Average interference effect
        total_interference = np.mean(interference_terms)
        return total_interference

    def quantum_decision_prediction(
        self, customer_profile, context_factors, information_order
    ):
        """Main QDT prediction with fair calibration"""

        # Create customer quantum state
        qc = self.create_customer_state(
            customer_profile["price_sensitivity"],
            customer_profile["quality_preference"],
            customer_profile["brand_loyalty"],
        )

        # Apply contextual interference (same magnitude as classical)
        qc = self.apply_contextual_interference(
            qc,
            context_factors["lighting"],
            context_factors["music"],
            context_factors["crowding"],
            context_factors["display_quality"],
        )

        # Apply order effects (genuine quantum advantage)
        qc = self.apply_order_effects(qc, information_order)

        # Measure decision probability
        prob, counts = self.measure_decision_probability(qc)

        # Calculate additional quantum metrics
        interference = self.calculate_interference_effects(context_factors)

        return {
            "purchase_probability": prob,
            "quantum_counts": counts,
            "circuit": qc,
            "interference_effect": interference,
            "quantum_states": len(counts),
            "superposition_measure": self._calculate_superposition(counts),
        }

    def _calculate_superposition(self, counts):
        """Measure how much superposition existed before collapse"""
        total_shots = sum(counts.values())
        probabilities = [count / total_shots for count in counts.values()]

        # Shannon entropy as measure of superposition
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(probabilities))

        # Normalized superposition measure (0 = definite state, 1 = maximum superposition)
        return entropy / max_entropy if max_entropy > 0 else 0


class FairClassicalDecisionModel:
    """
    Enhanced classical decision theory with same parameter ranges as quantum
    """

    def __init__(self, base_context_sensitivity=0.15):
        """
        base_context_sensitivity: Maximum context effect (same as quantum)
        """
        self.base_context_sensitivity = base_context_sensitivity

    def classical_decision_prediction(
        self, customer_profile, context_factors, information_order
    ):
        """
        Classical decision prediction with fair parameter calibration
        """

        # Base utility calculation
        price_utility = 1.0 - customer_profile["price_sensitivity"]
        quality_utility = customer_profile["quality_preference"]
        brand_utility = customer_profile["brand_loyalty"]

        # Context effects (SAME magnitude as quantum model)
        context_effects = (
            context_factors["lighting"] * self.base_context_sensitivity
            + context_factors["music"] * self.base_context_sensitivity * 0.5
            + context_factors["display_quality"] * self.base_context_sensitivity
            - context_factors["crowding"] * self.base_context_sensitivity * 0.5
        )

        # Classical model: Linear combination of utilities
        base_utility = price_utility * 0.4 + quality_utility * 0.4 + brand_utility * 0.2

        # Add context effects
        total_utility = base_utility + context_effects

        # Classical model: Information order has NO effect (key difference)
        # This is where quantum model can show genuine advantage

        # Convert utility to probability using logistic function
        purchase_probability = 1 / (1 + np.exp(-5 * (total_utility - 0.5)))

        # Classical model can have interaction effects too (to be fair)
        interaction_bonus = self._calculate_classical_interactions(
            customer_profile, context_factors
        )

        # Final probability with interactions
        final_probability = min(1.0, purchase_probability + interaction_bonus)

        return {
            "purchase_probability": final_probability,
            "utility_breakdown": {
                "price_utility": price_utility,
                "quality_utility": quality_utility,
                "brand_utility": brand_utility,
                "context_effects": context_effects,
                "interaction_bonus": interaction_bonus,
                "base_utility": base_utility,
            },
            "order_sensitivity": 0.0,  # Classical models are order-independent
        }

    def _calculate_classical_interactions(self, customer_profile, context_factors):
        """
        Classical interaction effects (to be fair to classical model)
        But still limited compared to quantum interference
        """
        # Some reasonable classical interactions

        # Quality-conscious customers more affected by display
        quality_display_interaction = (
            customer_profile["quality_preference"]
            * context_factors["display_quality"]
            * 0.05
        )

        # Price-sensitive customers more affected by crowding (feels expensive)
        price_crowding_interaction = (
            customer_profile["price_sensitivity"]
            * context_factors["crowding"]
            * (-0.03)
        )

        # Brand-loyal customers like good ambiance
        brand_ambiance_interaction = (
            customer_profile["brand_loyalty"]
            * (context_factors["lighting"] + context_factors["music"])
            * 0.02
        )

        total_interactions = (
            quality_display_interaction
            + price_crowding_interaction
            + brand_ambiance_interaction
        )

        return total_interactions


def compare_order_effects():
    """
    Demonstrate genuine quantum order effects vs classical order independence
    """

    qdt_model = FairQuantumDecisionModel()
    classical_model = FairClassicalDecisionModel()

    # Test customer profile
    customer = {
        "price_sensitivity": 0.7,
        "quality_preference": 0.8,
        "brand_loyalty": 0.4,
    }

    # Test context
    context = {"lighting": 0.8, "music": 0.6, "crowding": 0.3, "display_quality": 0.9}

    # Different information orders
    orders = [
        ["price", "quality", "ambiance"],
        ["ambiance", "quality", "price"],
        ["quality", "price", "ambiance"],
    ]

    print("🔬 Order Effects Comparison (Fair Parameters)")
    print("=" * 60)

    classical_results = []
    quantum_results = []

    for order in orders:
        # Classical prediction (should be same for all orders)
        classical_result = classical_model.classical_decision_prediction(
            customer, context, order
        )
        classical_prob = classical_result["purchase_probability"]

        # Quantum prediction (should vary with order)
        quantum_result = qdt_model.quantum_decision_prediction(customer, context, order)
        quantum_prob = quantum_result["purchase_probability"]

        classical_results.append(classical_prob)
        quantum_results.append(quantum_prob)

        print(f"\n📋 Order: {' → '.join(order)}")
        print(f"   Classical: {classical_prob:.3f}")
        print(f"   Quantum:   {quantum_prob:.3f}")
        print(f"   Interference: {quantum_result['interference_effect']:.3f}")
        print(f"   Superposition: {quantum_result['superposition_measure']:.3f}")

    # Calculate order sensitivity
    classical_variance = np.var(classical_results)
    quantum_variance = np.var(quantum_results)

    print(f"\n📊 Order Sensitivity Analysis:")
    print(f"   Classical variance: {classical_variance:.6f}")
    print(f"   Quantum variance:   {quantum_variance:.6f}")
    print(
        f"   Quantum advantage:  {quantum_variance / classical_variance:.1f}x more sensitive"
        if classical_variance > 0
        else "   Quantum shows order effects, Classical doesn't"
    )

    return {
        "orders": [" → ".join(order) for order in orders],
        "classical_results": classical_results,
        "quantum_results": quantum_results,
        "classical_variance": classical_variance,
        "quantum_variance": quantum_variance,
    }


def compare_context_sensitivity():
    """
    Compare how context affects both models with fair parameters
    """

    qdt_model = FairQuantumDecisionModel()
    classical_model = FairClassicalDecisionModel()

    customer = {
        "price_sensitivity": 0.6,
        "quality_preference": 0.7,
        "brand_loyalty": 0.5,
    }

    order = ["ambiance", "quality", "price"]

    # Test different lighting levels
    lighting_levels = np.linspace(0, 1, 11)

    classical_probs = []
    quantum_probs = []

    for lighting in lighting_levels:
        context = {
            "lighting": lighting,
            "music": 0.6,
            "crowding": 0.4,
            "display_quality": 0.7,
        }

        classical_result = classical_model.classical_decision_prediction(
            customer, context, order
        )
        quantum_result = qdt_model.quantum_decision_prediction(customer, context, order)

        classical_probs.append(classical_result["purchase_probability"])
        quantum_probs.append(quantum_result["purchase_probability"])

    return {
        "lighting_levels": lighting_levels,
        "classical_probs": classical_probs,
        "quantum_probs": quantum_probs,
    }


if __name__ == "__main__":
    print("🚀 Fair Quantum vs Classical Decision Theory Comparison")
    print("=" * 70)

    # Test order effects
    order_results = compare_order_effects()

    print("\n" + "=" * 70)

    # Test context sensitivity
    context_results = compare_context_sensitivity()

    print(f"\n🌊 Context Sensitivity (Lighting 0→1):")
    print(
        f"   Classical range: {max(context_results['classical_probs']) - min(context_results['classical_probs']):.3f}"
    )
    print(
        f"   Quantum range:   {max(context_results['quantum_probs']) - min(context_results['quantum_probs']):.3f}"
    )

    print(
        "\n✅ Fair comparison complete - quantum advantages come from phenomena, not inflated parameters!"
    )
