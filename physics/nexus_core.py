import taichi as ti
import math

@ti.data_oriented
class NexusCore:
    """
    Implements the fundamental physics constants and feedback loops 
    dictated by the Nexus Recursive Harmonic Architecture.
    """
    def __init__(self, res=512):
        self.RES = res
        
        # --- Component III: Constants & Tuning Parameters ---
        # The Mark 1 Attractor (M1): Universal Sweet Spot at the edge of chaos.
        # Approximately 0.318 (often related to 1/pi)
        self.MARK_1_ATTRACTOR = 0.318 
        
        # Track the current systemic state (deviation from M1)
        self.system_state = ti.field(dtype=float, shape=())
        self.system_state[None] = self.MARK_1_ATTRACTOR
        
        # --- Component IV: The Stroboscopic Universe & Time ---
        # Swapping Zero Mechanism: Time toggles between Noun/GR and Verb/QM phases
        # True = Noun phase (geometric/deterministic, structure preservation)
        # False = Verb phase (probabilistic/action, entropy shedding)
        self.is_noun_phase = ti.field(dtype=int, shape=())
        self.is_noun_phase[None] = 1 
        
        # Expansive Null (e) vs Curvature Null (Golden Ratio phi)
        self.EXPANSIVE_NULL = 2.71828
        self.CURVATURE_NULL = 1.61803
        
        # --- The Substrate (Alpha, Beta, Gamma) ---
        # Represented conceptually through the Pi-lattice interactions in GeodesicEngine

    @ti.kernel
    def step_time_stroboscopic(self):
        """
        Toggles the universal execution phase at the Planck frequency.
        """
        self.is_noun_phase[None] = 1 - self.is_noun_phase[None]

    @ti.kernel
    def apply_samsons_law(self, perturbation: float) -> float:
        """
        Component II: Samson's Law V2
        Acts as a universal feedback controller. The rate of change of the system 
        is proportional to the negative deviation from the Mark 1 Attractor.
        
        dX/dt = -k * (X - M1)
        """
        k = 0.05 # Feedback gain (stability factor)
        
        # Inject the external perturbation into the current state
        current_state = self.system_state[None] + perturbation
        
        # Calculate the deviation from the harmonic ideal
        deviation = current_state - self.MARK_1_ATTRACTOR
        
        # Apply the negative feedback correction
        correction = -k * deviation
        
        # Update the system state toward equilibrium
        new_state = current_state + correction
        
        # Clamp bounds to prevent absolute numerical explosion if perturbation is massive
        self.system_state[None] = ti.max(0.001, ti.min(1.0, new_state))
        
        # Return the stabilization factor to be used by other simulation modules
        # (e.g., dampening fluid vorticity or scaling L-system growth)
        return self.system_state[None] / self.MARK_1_ATTRACTOR

    def get_phase_multiplier(self) -> float:
        """
        Returns the appropriate mathematical constant based on the current Stroboscopic phase.
        Noun (Geometric) -> Curvature Null (Golden Ratio)
        Verb (Probabilistic) -> Expansive Null (Euler's Number)
        """
        if self.is_noun_phase[None] == 1:
            return self.CURVATURE_NULL
        else:
            return self.EXPANSIVE_NULL
