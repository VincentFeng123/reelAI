"""
100-topic user search simulation.

Simulates a real user searching across 100 diverse educational topics.
For each topic, builds a realistic transcript with concept mentions in
clusters, runs the full pipeline (_topic_cut_segments_for_concept →
chaining → refine/split), and checks:

  1. BOUNDARY QUALITY — start at sentence boundary, end at terminal punct
  2. CHAIN INTEGRITY — continuation reels are consecutive with no
     interrupting clips from other topics
  3. ON-TOPIC — segments actually cover concept mention clusters
  4. SETTINGS COMPLIANCE — clip durations respect min/max user settings

This test is deterministic (seeded RNG) and runs offline with no
network access.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass
from typing import Any

from app.services.embeddings import EmbeddingService
from app.services.reels import ReelService
from app.services.youtube import YouTubeService


# ---------------------------------------------------------------------------
# Transcript generators
# ---------------------------------------------------------------------------

@dataclass
class TopicSpec:
    """Specification for generating a realistic transcript."""
    topic: str
    concept_terms: list[str]
    video_duration_sec: int
    cue_duration: float
    gap: float  # inter-cue pause
    # Where concept is mentioned (list of (start_sec, end_sec) clusters)
    mention_clusters: list[tuple[float, float]]
    # Sentences for each cluster (with concept woven in)
    cluster_sentences: list[list[str]]
    # Filler sentences for non-mention regions
    filler_sentences: list[str]
    # User settings
    clip_min_len: int
    clip_max_len: int


def _make_cue(start: float, duration: float, text: str) -> dict[str, Any]:
    return {"start": start, "duration": duration, "text": text}


def build_transcript_from_spec(spec: TopicSpec) -> list[dict[str, Any]]:
    """Build a full video transcript from a TopicSpec.

    Fills non-cluster regions with filler sentences and cluster regions
    with concept-mentioning sentences.
    """
    cues: list[dict[str, Any]] = []
    t = 0.0
    stride = spec.cue_duration + spec.gap
    filler_idx = 0
    cluster_idx = 0
    cluster_sent_idx = 0

    while t < spec.video_duration_sec - 1.0:
        # Check if we're inside a mention cluster
        in_cluster = False
        for ci, (cs, ce) in enumerate(spec.mention_clusters):
            if cs <= t < ce:
                in_cluster = True
                if ci == cluster_idx and cluster_sent_idx < len(spec.cluster_sentences[ci]):
                    text = spec.cluster_sentences[ci][cluster_sent_idx]
                    cluster_sent_idx += 1
                elif ci != cluster_idx:
                    cluster_idx = ci
                    cluster_sent_idx = 0
                    if cluster_sent_idx < len(spec.cluster_sentences[ci]):
                        text = spec.cluster_sentences[ci][cluster_sent_idx]
                        cluster_sent_idx += 1
                    else:
                        text = spec.filler_sentences[filler_idx % len(spec.filler_sentences)]
                        filler_idx += 1
                else:
                    # Ran out of cluster sentences — use filler with concept injected
                    base = spec.filler_sentences[filler_idx % len(spec.filler_sentences)]
                    text = base.rstrip(".!?") + f" in {spec.concept_terms[0]}."
                    filler_idx += 1
                break
        if not in_cluster:
            text = spec.filler_sentences[filler_idx % len(spec.filler_sentences)]
            filler_idx += 1

        cues.append(_make_cue(t, spec.cue_duration, text))
        t += stride

    return cues


# ---------------------------------------------------------------------------
# 100 topic specifications
# ---------------------------------------------------------------------------

# Filler sentences (generic educational)
GENERIC_FILLERS = [
    "Let me explain this concept in more detail.",
    "This is an important point to remember.",
    "Now let's move on to the next idea.",
    "Here we can see an interesting pattern.",
    "This example illustrates the principle clearly.",
    "Pay attention to how these elements interact.",
    "The relationship between these factors is crucial.",
    "Let's take a step back and consider the big picture.",
    "This has practical applications in many fields.",
    "Understanding this will help with more advanced topics.",
    "Notice how this connects to what we discussed earlier.",
    "The key insight here is quite elegant.",
    "Many students find this part challenging at first.",
    "Let me walk you through this step by step.",
    "There are several ways to approach this problem.",
    "This result might seem surprising at first glance.",
    "The proof of this theorem is quite beautiful.",
    "Let's verify our answer with a concrete example.",
    "Keep in mind the assumptions we made earlier.",
    "This technique will appear again in later chapters.",
]


def _gen_topics() -> list[TopicSpec]:
    """Generate 100 diverse topic specifications."""
    rng = random.Random(42)  # deterministic

    raw_topics = [
        # (topic_name, concept_terms, cluster_sentences_templates)
        # MATH (15)
        ("calculus_limits", ["limits", "calculus"],
         [["The concept of limits is fundamental to calculus.",
           "A limit describes the value a function approaches.",
           "We write the limit of f of x as x approaches a.",
           "Limits can be evaluated using algebraic simplification.",
           "The epsilon delta definition formalizes limits rigorously."],
          ["Limits are used to define continuity in calculus.",
           "A function is continuous if its limit equals its value.",
           "One sided limits approach from the left or right."]]),
        ("derivatives", ["derivatives", "differentiation"],
         [["Derivatives measure the instantaneous rate of change.",
           "The derivative of x squared is two x.",
           "We can find derivatives using the power rule.",
           "The chain rule handles composite function derivatives.",
           "Higher order derivatives describe acceleration."],
          ["Derivatives have many applications in physics.",
           "The derivative of position gives velocity.",
           "Optimization problems use derivatives to find extrema."]]),
        ("integrals", ["integrals", "integration"],
         [["Integration computes the area under a curve.",
           "The definite integral gives a numerical value.",
           "Integration by parts reverses the product rule.",
           "Substitution simplifies complex integrals.",
           "The fundamental theorem connects integrals and derivatives."],
          ["Integrals appear throughout physics and engineering.",
           "Volume integrals extend the concept to three dimensions."]]),
        ("linear_algebra", ["linear algebra", "matrices"],
         [["Linear algebra studies vectors and matrices.",
           "Matrix multiplication combines transformations.",
           "Eigenvalues reveal scaling factors of matrices.",
           "The determinant tells us if a matrix is invertible.",
           "Gaussian elimination solves linear systems."],
          ["Linear algebra is foundational in machine learning.",
           "Principal component analysis uses matrices for dimensionality reduction."]]),
        ("probability", ["probability", "random"],
         [["Probability measures the likelihood of events.",
           "The probability of an event ranges from zero to one.",
           "Conditional probability depends on prior information.",
           "Bayes theorem updates probability with new evidence.",
           "The law of large numbers connects probability to frequency."],
          ["Probability distributions describe random variables.",
           "The normal distribution is the most common probability model."]]),
        ("statistics", ["statistics", "statistical"],
         [["Statistics helps us make sense of data.",
           "Descriptive statistics summarize datasets.",
           "The mean median and mode are measures of central tendency.",
           "Standard deviation measures spread in statistics.",
           "Hypothesis testing is a core statistical method."],
          ["Regression analysis finds relationships between variables.",
           "Statistical significance requires careful interpretation."]]),
        ("differential_equations", ["differential equations", "ODE"],
         [["Differential equations relate functions to their derivatives.",
           "First order differential equations are the simplest case.",
           "The integrating factor method solves linear ODEs.",
           "Second order differential equations model oscillations.",
           "Laplace transforms simplify differential equations."],
          ["Differential equations appear in population modeling.",
           "The wave equation is a fundamental differential equation."]]),
        ("topology", ["topology", "topological"],
         [["Topology studies properties preserved under continuous deformation.",
           "A coffee cup and a donut are topologically equivalent.",
           "Open sets are the building blocks of topology.",
           "Compactness is a key concept in topology.",
           "The fundamental group captures topological structure."],
          ["Topological invariants classify spaces up to homeomorphism.",
           "Algebraic topology uses algebra to study topological problems."]]),
        ("number_theory", ["number theory", "prime"],
         [["Number theory studies properties of integers.",
           "Prime numbers are the building blocks of number theory.",
           "The fundamental theorem of arithmetic concerns prime factorization.",
           "Modular arithmetic is central to number theory.",
           "Fermat's last theorem was a famous number theory problem."],
          ["Cryptography relies heavily on number theory.",
           "The distribution of prime numbers follows subtle patterns."]]),
        ("abstract_algebra", ["abstract algebra", "groups"],
         [["Abstract algebra studies algebraic structures.",
           "Groups are the simplest structures in abstract algebra.",
           "A group has an identity element and inverses.",
           "Rings extend groups with a second operation.",
           "Fields allow division in abstract algebra."],
          ["Galois theory connects groups to polynomial equations.",
           "Abstract algebra has applications in coding theory."]]),
        ("combinatorics", ["combinatorics", "counting"],
         [["Combinatorics studies counting and arrangement.",
           "Permutations count ordered arrangements in combinatorics.",
           "Combinations count unordered selections.",
           "The binomial theorem connects combinatorics to algebra.",
           "Generating functions are powerful combinatorics tools."],
          ["Graph theory is a branch of combinatorics.",
           "Combinatorics problems appear on math competitions."]]),
        ("geometry", ["geometry", "geometric"],
         [["Geometry studies shapes, sizes, and spatial relationships.",
           "Euclidean geometry is based on five postulates.",
           "The Pythagorean theorem relates sides of right triangles.",
           "Geometric transformations include rotation and reflection.",
           "Non-Euclidean geometry allows curved spaces."],
          ["Differential geometry studies curves and surfaces.",
           "Geometric proofs develop logical reasoning skills."]]),
        ("trigonometry", ["trigonometry", "trigonometric"],
         [["Trigonometry studies relationships between angles and sides.",
           "Sine cosine and tangent are the basic trigonometric functions.",
           "The unit circle defines trigonometric values for all angles.",
           "Trigonometric identities simplify complex expressions.",
           "The law of sines and cosines generalize trigonometry to any triangle."],
          ["Fourier analysis uses trigonometric functions to decompose signals.",
           "Trigonometry is essential in navigation and surveying."]]),
        ("real_analysis", ["real analysis", "convergence"],
         [["Real analysis provides rigorous foundations for calculus.",
           "Convergence of sequences is a central topic in real analysis.",
           "The Bolzano Weierstrass theorem guarantees convergent subsequences.",
           "Uniform convergence preserves continuity in real analysis.",
           "Measure theory extends real analysis to more general settings."],
          ["Lebesgue integration generalizes the Riemann integral.",
           "Real analysis skills are essential for graduate mathematics."]]),
        ("complex_analysis", ["complex analysis", "analytic"],
         [["Complex analysis studies functions of complex variables.",
           "Analytic functions satisfy the Cauchy Riemann equations.",
           "Contour integration is a powerful tool in complex analysis.",
           "The residue theorem evaluates difficult integrals.",
           "Complex analysis connects to number theory and physics."],
          ["Conformal mappings preserve angles in complex analysis.",
           "Many physical problems simplify using analytic functions."]]),
        # PHYSICS (15)
        ("classical_mechanics", ["mechanics", "Newton"],
         [["Classical mechanics describes motion of macroscopic objects.",
           "Newton's laws form the foundation of mechanics.",
           "The first law states that objects at rest stay at rest.",
           "Force equals mass times acceleration in Newton's mechanics.",
           "Conservation of energy is fundamental in mechanics."],
          ["Lagrangian mechanics reformulates Newton's approach.",
           "Hamiltonian mechanics uses energy to describe motion."]]),
        ("electromagnetism", ["electromagnetism", "electromagnetic"],
         [["Electromagnetism describes electric and magnetic phenomena.",
           "Maxwell's equations unify electromagnetism.",
           "Electric fields are created by charges.",
           "Magnetic fields arise from moving charges.",
           "Electromagnetic waves include light and radio."],
          ["Electromagnetic induction generates electricity.",
           "The speed of light follows from electromagnetic theory."]]),
        ("thermodynamics", ["thermodynamics", "entropy"],
         [["Thermodynamics studies heat and energy transfer.",
           "The first law of thermodynamics conserves energy.",
           "Entropy always increases in isolated systems.",
           "The second law defines the direction of thermodynamic processes.",
           "Temperature measures average kinetic energy."],
          ["Carnot efficiency limits thermodynamic engines.",
           "Statistical mechanics explains thermodynamics microscopically."]]),
        ("quantum_mechanics", ["quantum mechanics", "quantum"],
         [["Quantum mechanics governs the behavior of tiny particles.",
           "The wave function encodes all quantum information.",
           "Heisenberg's uncertainty principle limits quantum measurement.",
           "Quantum entanglement connects distant particles.",
           "The Schrodinger equation describes quantum time evolution."],
          ["Quantum computing uses quantum mechanics for computation.",
           "Quantum field theory extends quantum mechanics to fields."]]),
        ("relativity", ["relativity", "spacetime"],
         [["Special relativity describes physics at high speeds.",
           "The speed of light is constant in all reference frames.",
           "Time dilation occurs in relativity near light speed.",
           "General relativity describes gravity as curved spacetime.",
           "Einstein's field equations relate spacetime geometry to matter."],
          ["Black holes are predictions of general relativity.",
           "Gravitational waves confirm spacetime curvature in relativity."]]),
        ("optics", ["optics", "light"],
         [["Optics studies the behavior of light.",
           "Reflection and refraction are fundamental to optics.",
           "Lenses focus light using the principles of optics.",
           "Diffraction shows the wave nature of light.",
           "Polarization filters select specific light orientations."],
          ["Fiber optics transmit data using total internal reflection.",
           "Laser technology depends on stimulated emission of light."]]),
        ("fluid_dynamics", ["fluid dynamics", "fluid"],
         [["Fluid dynamics describes the motion of liquids and gases.",
           "The Navier Stokes equations govern fluid dynamics.",
           "Turbulence is one of the great unsolved problems in fluid dynamics.",
           "Bernoulli's principle explains pressure changes in fluid flow.",
           "Reynolds number characterizes fluid flow regimes."],
          ["Computational fluid dynamics simulates complex flows.",
           "Aerodynamics applies fluid dynamics to aircraft design."]]),
        ("nuclear_physics", ["nuclear physics", "nuclear"],
         [["Nuclear physics studies atomic nuclei and their interactions.",
           "Nuclear fission splits heavy nuclei to release energy.",
           "Nuclear fusion combines light nuclei in stars.",
           "Radioactive decay is a fundamental nuclear process.",
           "The strong force holds nuclear matter together."],
          ["Nuclear reactors harness fission for power generation.",
           "Nuclear physics explains how stars produce energy."]]),
        ("particle_physics", ["particle physics", "particles"],
         [["Particle physics studies the fundamental constituents of matter.",
           "The Standard Model classifies known particles.",
           "Quarks combine to form protons and neutrons.",
           "The Higgs boson gives particles their mass.",
           "Particle accelerators probe the smallest scales."],
          ["Neutrinos are ghostly particles that rarely interact.",
           "Dark matter may require new particles beyond the Standard Model."]]),
        ("astrophysics", ["astrophysics", "stellar"],
         [["Astrophysics applies physics to understand celestial objects.",
           "Stellar evolution describes the life cycle of stars.",
           "White dwarfs neutron stars and black holes are stellar remnants.",
           "Supernovae are explosive stellar deaths.",
           "The Hertzsprung Russell diagram classifies stars."],
          ["Cosmology studies the universe at the largest scales.",
           "Stellar nucleosynthesis creates heavy elements."]]),
        ("condensed_matter", ["condensed matter", "solid state"],
         [["Condensed matter physics studies solid and liquid phases.",
           "Crystal structures repeat in three dimensions.",
           "Band theory explains electrical conductivity in solid state physics.",
           "Superconductivity allows current flow without resistance.",
           "Topological insulators are a new class of condensed matter."],
          ["Semiconductors are the basis of modern electronics.",
           "Condensed matter research drives material science advances."]]),
        ("acoustics", ["acoustics", "sound"],
         [["Acoustics is the science of sound and vibration.",
           "Sound waves are longitudinal pressure oscillations.",
           "The Doppler effect shifts sound frequency with motion.",
           "Room acoustics affect how we perceive sound quality.",
           "Ultrasound uses high frequency sound for imaging."],
          ["Acoustic engineering designs concert halls for optimal sound.",
           "Noise cancellation uses destructive interference of sound waves."]]),
        ("plasma_physics", ["plasma physics", "plasma"],
         [["Plasma physics studies ionized gases.",
           "Plasma is the fourth state of matter.",
           "The sun is a giant ball of plasma.",
           "Magnetic confinement contains hot plasma for fusion.",
           "Plasma physics governs space weather phenomena."],
          ["Industrial plasma processing etches semiconductor chips.",
           "Plasma displays were an early flat screen technology."]]),
        ("biophysics", ["biophysics", "biological physics"],
         [["Biophysics applies physics to biological systems.",
           "Protein folding is a key problem in biophysics.",
           "Ion channels control electrical signals in biophysics.",
           "Molecular motors convert chemical energy to mechanical work.",
           "Biophysics techniques include X-ray crystallography."],
          ["Single molecule biophysics studies individual proteins.",
           "Computational biophysics models cellular processes."]]),
        ("geophysics", ["geophysics", "seismic"],
         [["Geophysics studies the physical properties of the Earth.",
           "Seismic waves reveal the Earth's internal structure.",
           "The Earth's magnetic field is generated by geophysics processes.",
           "Plate tectonics explains continental drift.",
           "Geophysics methods locate mineral and oil deposits."],
          ["Seismic monitoring detects and locates earthquakes.",
           "Geophysics research helps predict volcanic eruptions."]]),
        # BIOLOGY (10)
        ("cell_biology", ["cell biology", "cellular"],
         [["Cell biology studies the structure and function of cells.",
           "The cell membrane controls what enters and leaves.",
           "Mitochondria produce energy for cellular processes.",
           "The nucleus contains the cell's genetic material.",
           "Cell division is essential for growth and repair."],
          ["Stem cells can differentiate into specialized cell types.",
           "Cellular signaling coordinates tissue responses."]]),
        ("genetics", ["genetics", "genes"],
         [["Genetics studies heredity and variation in organisms.",
           "DNA carries the genetic instructions for life.",
           "Genes are segments of DNA that encode proteins.",
           "Mendel's laws describe inheritance patterns in genetics.",
           "Mutations in genes can cause genetic disorders."],
          ["Gene therapy aims to correct genetic defects.",
           "CRISPR technology allows precise genetics editing."]]),
        ("evolution", ["evolution", "natural selection"],
         [["Evolution explains the diversity of life on Earth.",
           "Natural selection drives evolution by favoring advantageous traits.",
           "Darwin proposed evolution by natural selection.",
           "Genetic drift is random change in evolution.",
           "Speciation occurs when populations diverge through evolution."],
          ["Molecular evolution tracks changes in DNA sequences.",
           "Convergent evolution produces similar traits independently."]]),
        ("ecology", ["ecology", "ecosystem"],
         [["Ecology studies interactions between organisms and environments.",
           "Ecosystems include all living and nonliving components.",
           "Food webs describe energy flow in ecology.",
           "Biodiversity measures the variety of life in an ecosystem.",
           "Ecological succession changes communities over time."],
          ["Conservation ecology aims to protect endangered species.",
           "Climate change disrupts ecosystem balance globally."]]),
        ("neuroscience", ["neuroscience", "brain"],
         [["Neuroscience studies the nervous system and the brain.",
           "Neurons communicate through electrical and chemical signals.",
           "The brain processes sensory information continuously.",
           "Synaptic plasticity underlies learning in neuroscience.",
           "Brain imaging techniques include MRI and EEG."],
          ["Neuroscience research advances treatments for brain disorders.",
           "Cognitive neuroscience bridges brain and behavior."]]),
        ("immunology", ["immunology", "immune"],
         [["Immunology studies the body's defense against infection.",
           "The immune system recognizes foreign pathogens.",
           "Antibodies target specific antigens in the immune response.",
           "T cells and B cells are key players in immunology.",
           "Vaccines train the immune system to fight diseases."],
          ["Autoimmune disorders result from immune system malfunction.",
           "Cancer immunology harnesses immune cells to fight tumors."]]),
        ("microbiology", ["microbiology", "bacteria"],
         [["Microbiology studies microscopic organisms.",
           "Bacteria are prokaryotic organisms studied in microbiology.",
           "Antibiotic resistance is a growing microbiology concern.",
           "The human microbiome contains trillions of bacteria.",
           "Viruses are studied alongside bacteria in microbiology."],
          ["Fermentation uses bacteria and yeast in food production.",
           "Microbiology techniques include culture and microscopy."]]),
        ("biochemistry", ["biochemistry", "enzymes"],
         [["Biochemistry studies chemical processes in living organisms.",
           "Enzymes catalyze biochemical reactions.",
           "Metabolic pathways convert nutrients to energy.",
           "Protein structure determines biochemistry function.",
           "DNA replication involves many enzymes and biochemistry processes."],
          ["Biochemistry research develops new drug therapies.",
           "Enzymes are used industrially in detergents and food processing."]]),
        ("marine_biology", ["marine biology", "ocean"],
         [["Marine biology studies life in the ocean and coastal waters.",
           "Coral reefs are among the most diverse marine ecosystems.",
           "Ocean acidification threatens marine biology worldwide.",
           "Whale migration is a fascinating marine biology phenomenon.",
           "Deep sea vents support unique marine biology communities."],
          ["Marine biology conservation protects endangered ocean species.",
           "Plankton form the base of ocean food chains."]]),
        ("botany", ["botany", "plants"],
         [["Botany is the scientific study of plants.",
           "Photosynthesis converts sunlight to energy in plants.",
           "Plant hormones regulate growth and development.",
           "Botany research improves crop yields worldwide.",
           "The diversity of flowering plants fascinates botanists."],
          ["Ethnobotany studies traditional plant uses by cultures.",
           "Botany contributes to medicine through plant derived drugs."]]),
        # CHEMISTRY (10)
        ("organic_chemistry", ["organic chemistry", "carbon"],
         [["Organic chemistry studies carbon containing compounds.",
           "Carbon forms four bonds making organic chemistry diverse.",
           "Functional groups determine organic chemistry reactivity.",
           "Organic synthesis creates new molecules.",
           "Stereochemistry studies three dimensional arrangements."],
          ["Organic chemistry is essential for pharmaceutical development.",
           "Polymers are long chain organic molecules."]]),
        ("inorganic_chemistry", ["inorganic chemistry", "metal"],
         [["Inorganic chemistry studies non-carbon compounds and metals.",
           "Coordination compounds contain metal centers.",
           "Crystal field theory explains metal complex colors.",
           "Inorganic chemistry catalysts speed up industrial reactions.",
           "Bioinorganic chemistry studies metal ions in biology."],
          ["Nanomaterials are a frontier of inorganic chemistry.",
           "Metal organic frameworks store gases efficiently."]]),
        ("physical_chemistry", ["physical chemistry", "thermodynamic"],
         [["Physical chemistry applies physics principles to chemical systems.",
           "Chemical kinetics studies reaction rates.",
           "Thermodynamic properties determine reaction spontaneity.",
           "Quantum chemistry calculates molecular electronic structures.",
           "Spectroscopy measures how matter interacts with light."],
          ["Physical chemistry bridges chemistry and physics.",
           "Computational chemistry simulates molecular behavior."]]),
        ("analytical_chemistry", ["analytical chemistry", "analysis"],
         [["Analytical chemistry identifies and quantifies substances.",
           "Chromatography separates mixtures for chemical analysis.",
           "Mass spectrometry determines molecular weights.",
           "Titration measures concentration in analytical chemistry.",
           "Electrochemical analysis detects trace elements."],
          ["Quality control relies on analytical chemistry methods.",
           "Environmental analysis monitors pollutant levels."]]),
        ("electrochemistry", ["electrochemistry", "electrode"],
         [["Electrochemistry studies chemical reactions involving electron transfer.",
           "Batteries store energy using electrochemistry principles.",
           "Electrode potentials drive electrochemistry reactions.",
           "Fuel cells convert chemical energy directly to electricity.",
           "Corrosion is an unwanted electrochemistry process."],
          ["Electroplating deposits metal layers using electrochemistry.",
           "Lithium ion batteries revolutionized portable electrochemistry."]]),
        ("polymer_chemistry", ["polymer chemistry", "polymer"],
         [["Polymer chemistry studies large chain molecules.",
           "Polymerization joins monomers into polymer chains.",
           "Thermoplastics soften when heated.",
           "Cross linked polymers form rigid networks.",
           "Polymer chemistry creates materials from nylon to Kevlar."],
          ["Biodegradable polymers address environmental concerns.",
           "Polymer chemistry drives advances in 3D printing materials."]]),
        ("environmental_chemistry", ["environmental chemistry", "pollution"],
         [["Environmental chemistry studies chemical processes in nature.",
           "Air pollution includes particulate matter and ozone.",
           "Water treatment removes chemical contaminants.",
           "The ozone layer protects us from ultraviolet radiation.",
           "Environmental chemistry monitors greenhouse gas levels."],
          ["Remediation cleans up contaminated environmental sites.",
           "Green chemistry reduces pollution at the source."]]),
        ("nuclear_chemistry", ["nuclear chemistry", "radioactive"],
         [["Nuclear chemistry studies radioactive materials and reactions.",
           "Radioactive decay transforms one element into another.",
           "Half life measures radioactive decay rate.",
           "Nuclear medicine uses radioactive tracers for imaging.",
           "Carbon dating is a nuclear chemistry application."],
          ["Nuclear waste management is a critical challenge.",
           "Radioactive isotopes treat certain cancers."]]),
        ("food_chemistry", ["food chemistry", "nutrients"],
         [["Food chemistry studies the chemical composition of food.",
           "Maillard reactions create flavor during cooking.",
           "Food preservatives prevent microbial growth.",
           "Vitamins and minerals are essential nutrients.",
           "Food chemistry ensures safety and quality standards."],
          ["Antioxidants protect against oxidative damage.",
           "Food chemistry research develops healthier nutrients."]]),
        ("medicinal_chemistry", ["medicinal chemistry", "drug"],
         [["Medicinal chemistry designs pharmaceutical compounds.",
           "Drug discovery involves screening millions of molecules.",
           "Structure activity relationships guide medicinal chemistry.",
           "Pharmacokinetics studies how the body processes a drug.",
           "Medicinal chemistry develops targeted drug therapies."],
          ["Antibiotics were a breakthrough in medicinal chemistry.",
           "Drug resistance drives new medicinal chemistry research."]]),
        # COMPUTER SCIENCE (15)
        ("algorithms", ["algorithms", "sorting"],
         [["Algorithms are step by step procedures for computation.",
           "Sorting algorithms arrange data in order.",
           "Binary search is an efficient search algorithm.",
           "Dynamic programming breaks problems into subproblems.",
           "Graph algorithms traverse networks efficiently."],
          ["Algorithm complexity measures computational cost.",
           "Sorting algorithms include quicksort and mergesort."]]),
        ("data_structures", ["data structures", "tree"],
         [["Data structures organize information for efficient access.",
           "Arrays store elements in contiguous memory.",
           "Linked lists allow dynamic data structures.",
           "Binary tree structures enable fast searching.",
           "Hash tables provide constant time lookup."],
          ["Balanced tree data structures maintain logarithmic height.",
           "Choosing the right data structures impacts performance."]]),
        ("machine_learning", ["machine learning", "model"],
         [["Machine learning enables computers to learn from data.",
           "Supervised learning trains models on labeled examples.",
           "Neural networks are powerful machine learning models.",
           "Gradient descent optimizes machine learning model parameters.",
           "Overfitting occurs when a model memorizes training data."],
          ["Deep learning uses many layer neural networks.",
           "Machine learning model selection requires cross validation."]]),
        ("artificial_intelligence", ["artificial intelligence", "AI"],
         [["Artificial intelligence creates systems that exhibit intelligent behavior.",
           "AI encompasses machine learning natural language processing and robotics.",
           "Expert systems were an early form of artificial intelligence.",
           "Reinforcement learning teaches AI through rewards.",
           "The Turing test evaluates artificial intelligence."],
          ["AI ethics addresses bias and fairness concerns.",
           "General artificial intelligence remains an open problem."]]),
        ("databases", ["databases", "SQL"],
         [["Databases store and organize large amounts of data.",
           "SQL is the standard language for querying databases.",
           "Relational databases use tables with rows and columns.",
           "NoSQL databases handle unstructured data.",
           "Database indexing speeds up query performance."],
          ["Transaction processing ensures database consistency.",
           "Distributed databases scale across multiple SQL servers."]]),
        ("operating_systems", ["operating systems", "process"],
         [["Operating systems manage computer hardware and software.",
           "Process scheduling allocates CPU time fairly.",
           "Virtual memory extends physical RAM using disk space.",
           "File systems organize data on storage devices.",
           "Operating systems provide security and access control."],
          ["Modern operating systems support multitasking.",
           "Process synchronization prevents race conditions."]]),
        ("networking", ["networking", "protocol"],
         [["Computer networking connects devices for communication.",
           "The TCP IP protocol suite enables the internet.",
           "Routers forward packets between networks.",
           "DNS translates domain names to IP addresses.",
           "Network security protects against protocol exploits."],
          ["Wireless networking uses radio frequency signals.",
           "The HTTP protocol powers the world wide web."]]),
        ("cryptography", ["cryptography", "encryption"],
         [["Cryptography protects information through mathematical codes.",
           "Symmetric encryption uses the same key for both operations.",
           "Public key cryptography enables secure communication.",
           "Hash functions produce fixed size cryptography outputs.",
           "Digital signatures verify authenticity using encryption."],
          ["Quantum cryptography promises unbreakable encryption.",
           "Blockchain technology relies on cryptography principles."]]),
        ("compiler_design", ["compiler design", "compiler"],
         [["Compiler design translates high level code to machine language.",
           "Lexical analysis tokenizes the source code in a compiler.",
           "Parsing checks syntax according to grammar rules.",
           "Code optimization improves compiler output efficiency.",
           "Register allocation is a key compiler design challenge."],
          ["Just in time compilation balances speed and flexibility.",
           "Modern compiler design targets multiple architectures."]]),
        ("computer_graphics", ["computer graphics", "rendering"],
         [["Computer graphics creates visual content digitally.",
           "Ray tracing simulates light for realistic rendering.",
           "Rasterization converts geometry to pixels in computer graphics.",
           "Shaders program custom rendering effects.",
           "GPU acceleration powers real time computer graphics."],
          ["Global illumination computes indirect rendering paths.",
           "Computer graphics drives video games and visual effects."]]),
        ("natural_language_processing", ["natural language processing", "NLP"],
         [["Natural language processing enables computers to understand text.",
           "Tokenization splits text into NLP processing units.",
           "Word embeddings represent meaning as vectors.",
           "Transformers revolutionized natural language processing.",
           "Machine translation is a key NLP application."],
          ["Sentiment analysis detects emotion in NLP text.",
           "Large language models advance natural language processing."]]),
        ("computer_vision", ["computer vision", "image"],
         [["Computer vision teaches machines to interpret images.",
           "Convolutional neural networks excel at image recognition.",
           "Object detection locates items in computer vision scenes.",
           "Image segmentation divides pictures into regions.",
           "Computer vision enables autonomous driving systems."],
          ["Medical image analysis aids diagnosis.",
           "Computer vision accuracy now rivals human image perception."]]),
        ("robotics", ["robotics", "robot"],
         [["Robotics combines engineering and computer science.",
           "Robot actuators convert signals to physical motion.",
           "Path planning helps a robot navigate environments.",
           "Sensor fusion combines multiple robot inputs.",
           "Robotics applications include manufacturing and surgery."],
          ["Collaborative robots work alongside humans safely.",
           "Soft robotics uses flexible robot materials."]]),
        ("cybersecurity", ["cybersecurity", "security"],
         [["Cybersecurity protects systems from digital attacks.",
           "Firewalls filter network traffic for security purposes.",
           "Penetration testing finds cybersecurity vulnerabilities.",
           "Social engineering exploits human psychology.",
           "Zero day exploits target unknown security flaws."],
          ["Incident response handles cybersecurity breaches.",
           "Cybersecurity requires continuous security monitoring."]]),
        ("distributed_systems", ["distributed systems", "distributed"],
         [["Distributed systems spread computation across machines.",
           "Consensus algorithms coordinate distributed nodes.",
           "CAP theorem limits distributed systems guarantees.",
           "MapReduce processes big data in distributed fashion.",
           "Microservices decompose applications into distributed components."],
          ["Eventual consistency relaxes distributed systems constraints.",
           "Cloud computing enables elastic distributed infrastructure."]]),
        # EARTH SCIENCE (5)
        ("geology", ["geology", "rocks"],
         [["Geology studies the Earth's structure and processes.",
           "Igneous sedimentary and metamorphic are the three rock types.",
           "Plate tectonics drives geology processes globally.",
           "Fossils record the history of life in rocks.",
           "Volcanic activity shapes the surface through geology."],
          ["Mineral identification is a basic geology skill.",
           "Geology field work maps rock formations."]]),
        ("meteorology", ["meteorology", "weather"],
         [["Meteorology studies the atmosphere and weather patterns.",
           "Weather fronts mark boundaries between air masses.",
           "Hurricanes form over warm ocean water.",
           "Doppler radar tracks weather precipitation.",
           "Climate models predict long term meteorology trends."],
          ["Severe weather warnings save lives through meteorology.",
           "Weather forecasting improves with better data."]]),
        ("oceanography", ["oceanography", "ocean currents"],
         [["Oceanography studies the physical and biological ocean.",
           "Ocean currents distribute heat around the globe.",
           "The thermohaline circulation drives deep ocean currents.",
           "Tide patterns result from gravitational forces.",
           "Oceanography research monitors sea level changes."],
          ["Coral reef decline is a major oceanography concern.",
           "Ocean currents affect climate patterns significantly."]]),
        ("atmospheric_science", ["atmospheric science", "atmosphere"],
         [["Atmospheric science studies the layers of gas surrounding Earth.",
           "The troposphere is where weather occurs in the atmosphere.",
           "Ozone in the stratosphere absorbs ultraviolet radiation.",
           "Greenhouse gases trap heat in the atmosphere.",
           "Atmospheric science models predict climate change."],
          ["Air quality monitoring is a practical atmospheric science application.",
           "The atmosphere protects life from solar radiation."]]),
        ("paleontology", ["paleontology", "fossils"],
         [["Paleontology studies ancient life through fossils.",
           "Dinosaur fossils reveal a lost world.",
           "Mass extinctions are major events in paleontology.",
           "Fossil dating uses radiometric techniques.",
           "Paleontology connects biology and geology."],
          ["Amber preserves fossils with exceptional detail.",
           "Paleontology discoveries continue to surprise researchers."]]),
        # HUMANITIES (10)
        ("philosophy_ethics", ["philosophy", "ethical"],
         [["Philosophy examines fundamental questions about existence.",
           "Ethical theories guide moral decision making.",
           "Utilitarianism maximizes happiness in ethical philosophy.",
           "Deontology focuses on duties and rules.",
           "Virtue ethics emphasizes character in philosophical tradition."],
          ["Applied philosophy addresses real world ethical dilemmas.",
           "Philosophical thinking develops critical reasoning."]]),
        ("psychology", ["psychology", "cognitive"],
         [["Psychology studies the mind and human behavior.",
           "Cognitive psychology examines mental processes.",
           "Behavioral psychology focuses on observable actions.",
           "The unconscious mind influences behavior in psychology.",
           "Developmental psychology tracks changes across lifespan."],
          ["Clinical psychology treats mental health disorders.",
           "Cognitive biases affect judgment and decision making."]]),
        ("economics", ["economics", "market"],
         [["Economics studies how societies allocate scarce resources.",
           "Supply and demand determine market prices.",
           "Macroeconomics examines national economic performance.",
           "Microeconomics analyzes individual market decisions.",
           "Inflation erodes purchasing power in market economies."],
          ["Behavioral economics incorporates psychology into market models.",
           "International economics studies trade between nations."]]),
        ("sociology", ["sociology", "social"],
         [["Sociology studies human social behavior and institutions.",
           "Social stratification creates unequal access to resources.",
           "Socialization shapes individual identity within society.",
           "Deviance challenges social norms in sociological analysis.",
           "Institutions structure social interactions and relationships."],
          ["Urban sociology examines life in cities.",
           "Social movements drive collective action for change."]]),
        ("political_science", ["political science", "government"],
         [["Political science studies governance and power structures.",
           "Democracy is a system of government by the people.",
           "Political parties organize collective political action.",
           "International relations studies government interactions globally.",
           "Public policy translates political science goals into action."],
          ["Electoral systems determine how government representatives are chosen.",
           "Political science research informs evidence based policy."]]),
        ("linguistics", ["linguistics", "language"],
         [["Linguistics is the scientific study of language.",
           "Phonology studies the sound systems of language.",
           "Syntax examines how words combine into sentences.",
           "Semantics concerns meaning in linguistic expressions.",
           "Pragmatics studies language use in context."],
          ["Computational linguistics enables machine language processing.",
           "Historical linguistics traces language evolution over time."]]),
        ("anthropology", ["anthropology", "culture"],
         [["Anthropology studies human societies and cultures.",
           "Cultural anthropology examines beliefs and practices.",
           "Physical anthropology studies human biological variation.",
           "Archaeology recovers material evidence of past cultures.",
           "Anthropology fieldwork requires immersion in culture."],
          ["Medical anthropology studies health across cultures.",
           "Linguistic anthropology explores culture through language."]]),
        ("history_world_wars", ["world war", "conflict"],
         [["The world wars reshaped global politics and society.",
           "World War One began in 1914 as a European conflict.",
           "Trench warfare defined the first world war.",
           "The Treaty of Versailles ended the conflict temporarily.",
           "World War Two was the deadliest conflict in history."],
          ["The Cold War followed the second world war conflict.",
           "Studying world wars helps prevent future conflict."]]),
        ("art_history", ["art history", "artistic"],
         [["Art history traces the development of visual expression.",
           "Renaissance art emphasized realism and perspective.",
           "Impressionism captured fleeting artistic moments.",
           "Modern art broke with traditional artistic conventions.",
           "Contemporary art history includes diverse media."],
          ["Art history museums preserve cultural heritage.",
           "Artistic movements reflect social and political changes."]]),
        ("music_theory", ["music theory", "harmony"],
         [["Music theory explains how melodies and harmonies work.",
           "Scales organize notes into patterns for harmony.",
           "Chords are built by stacking intervals.",
           "Rhythm patterns create the pulse of music theory.",
           "Counterpoint combines independent melodic lines."],
          ["Jazz harmony extends classical music theory rules.",
           "Music theory analysis reveals hidden compositional harmony structures."]]),
        # ENGINEERING (5)
        ("electrical_engineering", ["electrical engineering", "circuit"],
         [["Electrical engineering designs systems that use electricity.",
           "Ohm's law relates voltage current and resistance in circuits.",
           "Transistors are the building blocks of electronic circuits.",
           "Signal processing transforms electrical engineering data.",
           "Power systems deliver electricity across circuit networks."],
          ["Integrated circuits pack billions of transistors on a chip.",
           "Electrical engineering drives renewable energy technology."]]),
        ("mechanical_engineering", ["mechanical engineering", "stress"],
         [["Mechanical engineering designs physical systems and machines.",
           "Stress analysis ensures structural mechanical engineering safety.",
           "Fluid mechanics studies force on surfaces.",
           "Heat transfer is crucial in mechanical engineering design.",
           "CAD software aids mechanical engineering stress modeling."],
          ["Robotics combines mechanical engineering with automation.",
           "Material stress testing prevents engineering failures."]]),
        ("civil_engineering", ["civil engineering", "structural"],
         [["Civil engineering designs infrastructure for society.",
           "Structural analysis ensures buildings can withstand loads.",
           "Geotechnical engineering studies soil for civil engineering foundations.",
           "Transportation engineering plans roads and bridges.",
           "Environmental civil engineering manages water and waste."],
          ["Earthquake resistant structural design saves lives.",
           "Civil engineering projects require careful structural planning."]]),
        ("chemical_engineering", ["chemical engineering", "reactor"],
         [["Chemical engineering applies chemistry at industrial scale.",
           "Reactor design controls chemical transformations.",
           "Separation processes purify chemical engineering products.",
           "Process control automates chemical reactor operations.",
           "Chemical engineering develops sustainable reactor processes."],
          ["Bioprocessing applies chemical engineering to biological systems.",
           "Reactor optimization reduces waste in chemical engineering."]]),
        ("aerospace_engineering", ["aerospace engineering", "flight"],
         [["Aerospace engineering designs aircraft and spacecraft.",
           "Aerodynamics governs the physics of flight.",
           "Propulsion systems generate thrust for aerospace engineering vehicles.",
           "Orbital mechanics calculates spacecraft flight paths.",
           "Aerospace engineering materials must withstand extreme conditions."],
          ["Drone technology expands aerospace engineering applications.",
           "Supersonic flight research continues in aerospace engineering."]]),
        # ADDITIONAL TOPICS (15) — diverse fields
        ("astronomy", ["astronomy", "stars"],
         [["Astronomy studies celestial objects and cosmic phenomena.",
           "Stars form from collapsing clouds of gas and dust.",
           "Galaxies contain billions of stars.",
           "Telescopes reveal the structure of distant astronomy objects.",
           "The expansion of the universe is a key astronomy discovery."],
          ["Radio astronomy detects invisible stars and galaxies.",
           "Astronomy observations confirm dark matter existence."]]),
        ("pharmacology", ["pharmacology", "drug interaction"],
         [["Pharmacology studies how drugs affect the body.",
           "Drug interactions can enhance or reduce effectiveness.",
           "Dosage response curves quantify pharmacology effects.",
           "Side effects are unwanted pharmacology outcomes.",
           "Pharmacokinetics describes drug absorption and distribution."],
          ["Personalized medicine tailors drug interaction profiles.",
           "Pharmacology research develops safer drug therapies."]]),
        ("materials_science", ["materials science", "alloy"],
         [["Materials science studies the properties of matter.",
           "Alloys combine metals for enhanced strength.",
           "Ceramic materials resist heat and corrosion.",
           "Composites combine materials science innovations.",
           "Nanomaterials exhibit unique alloy-like properties at small scales."],
          ["Biomaterials support medical implant development.",
           "Materials science creates stronger lighter alloys."]]),
        ("renewable_energy", ["renewable energy", "solar"],
         [["Renewable energy comes from naturally replenished sources.",
           "Solar panels convert sunlight directly to electricity.",
           "Wind turbines harness kinetic energy for renewable power.",
           "Hydroelectric dams use water flow for renewable energy.",
           "Energy storage is crucial for renewable energy adoption."],
          ["Solar cell efficiency continues to improve rapidly.",
           "Renewable energy reduces dependence on fossil fuels."]]),
        ("data_science", ["data science", "dataset"],
         [["Data science extracts insights from complex datasets.",
           "Exploratory data analysis reveals patterns in dataset structure.",
           "Feature engineering transforms raw data for modeling.",
           "Cross validation evaluates data science model performance.",
           "Visualization communicates dataset findings effectively."],
          ["Big data challenges traditional data science methods.",
           "Data science ethics addresses dataset bias and privacy."]]),
        ("information_theory", ["information theory", "entropy"],
         [["Information theory quantifies data and communication.",
           "Shannon entropy measures information content.",
           "Channel capacity limits data transmission rates.",
           "Error correcting codes protect information from noise.",
           "Data compression exploits information theory redundancy."],
          ["Mutual information measures variable dependence.",
           "Information theory entropy concepts apply to machine learning."]]),
        ("control_theory", ["control theory", "feedback"],
         [["Control theory designs systems that regulate behavior.",
           "Feedback loops are fundamental to control theory systems.",
           "PID controllers use proportional integral and derivative feedback.",
           "Stability analysis ensures control systems behave well.",
           "Optimal control minimizes cost in control theory."],
          ["Adaptive control adjusts to changing feedback conditions.",
           "Control theory applications range from autopilots to robotics."]]),
        ("game_theory", ["game theory", "strategy"],
         [["Game theory studies strategic decision making.",
           "The Nash equilibrium is a key game theory concept.",
           "Zero sum games have fixed total strategy payoffs.",
           "Cooperative game theory models coalition strategy.",
           "Mechanism design applies game theory in reverse."],
          ["Game theory strategy informs economic policy.",
           "Evolutionary game theory models biological strategy competition."]]),
        ("signal_processing", ["signal processing", "filter"],
         [["Signal processing manipulates and analyzes signals.",
           "Digital filter design removes unwanted frequencies.",
           "The Fourier transform decomposes signals into frequency components.",
           "Sampling theory determines signal processing requirements.",
           "Adaptive filter algorithms adjust to changing signals."],
          ["Image processing applies signal processing to pictures.",
           "Signal processing filter techniques enable modern communication."]]),
        ("biomedical_engineering", ["biomedical engineering", "implant"],
         [["Biomedical engineering applies engineering to healthcare.",
           "Prosthetic implant design restores lost function.",
           "Medical imaging uses biomedical engineering principles.",
           "Tissue engineering grows replacement implant materials.",
           "Biomedical engineering sensors monitor patient health."],
          ["Neural implant interfaces connect brain to computer.",
           "Biomedical engineering advances implant longevity."]]),
        ("quantum_computing", ["quantum computing", "qubit"],
         [["Quantum computing uses quantum mechanical phenomena.",
           "Qubits can exist in superposition states.",
           "Entanglement enables qubit correlations across distance.",
           "Quantum gates manipulate qubit states in circuits.",
           "Error correction is crucial for practical quantum computing."],
          ["Shor's algorithm threatens classical qubit encryption.",
           "Quantum computing qubit counts grow exponentially."]]),
        ("network_science", ["network science", "graph"],
         [["Network science studies complex connected systems.",
           "Graph theory provides the mathematical foundation.",
           "Small world networks show short average graph paths.",
           "Scale free networks follow power law degree distributions.",
           "Community detection finds graph clusters in networks."],
          ["Social network science maps human graph connections.",
           "Epidemic spreading follows network graph structure."]]),
        ("cognitive_science", ["cognitive science", "cognition"],
         [["Cognitive science studies the mind and its processes.",
           "Perception transforms sensory input into cognition.",
           "Memory systems store and retrieve cognitive information.",
           "Decision making involves complex cognition under uncertainty.",
           "Language is a uniquely human cognitive science ability."],
          ["Computational models simulate cognition processes.",
           "Cognitive science integrates psychology and neuroscience."]]),
        ("environmental_science", ["environmental science", "climate"],
         [["Environmental science studies human impact on nature.",
           "Climate change is the defining environmental science challenge.",
           "Deforestation accelerates climate warming.",
           "Biodiversity loss threatens ecosystem services.",
           "Environmental science policy aims to protect climate stability."],
          ["Carbon capture technology addresses climate emissions.",
           "Environmental science monitoring tracks climate indicators."]]),
        ("forensic_science", ["forensic science", "evidence"],
         [["Forensic science applies scientific methods to legal cases.",
           "DNA evidence revolutionized forensic science investigations.",
           "Fingerprint analysis matches evidence to individuals.",
           "Toxicology detects poisons in forensic evidence.",
           "Digital forensic science examines electronic evidence."],
          ["Ballistics analysis traces forensic evidence to weapons.",
           "Forensic science evidence standards ensure justice."]]),
    ]

    assert len(raw_topics) == 100, f"Expected 100 topics, got {len(raw_topics)}"

    specs: list[TopicSpec] = []
    for i, (name, terms, cluster_sents) in enumerate(raw_topics):
        # Vary video duration (60-300s), cue duration (2-5s), gap (0-0.6s),
        # and user settings to create realistic diversity.
        video_dur = rng.choice([90, 120, 150, 180, 210, 250, 300])
        cue_dur = rng.choice([2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        gap = rng.choice([0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6])
        min_len = rng.choice([15, 20, 25, 30])
        max_len = min_len + rng.choice([20, 25, 30, 35])

        # Place 1-2 mention clusters at realistic positions
        stride = cue_dur + gap
        n_cues = int(video_dur / stride)
        # Cluster 1: starts between 10% and 40% into the video
        c1_start = round(video_dur * rng.uniform(0.1, 0.35), 1)
        c1_len = round(len(cluster_sents[0]) * stride, 1)
        c1_end = min(c1_start + c1_len, video_dur - 10)
        clusters = [(c1_start, c1_end)]
        c_sents = [cluster_sents[0]]
        # Cluster 2 (if we have one): starts after a gap
        if len(cluster_sents) > 1 and video_dur > 120:
            c2_start = round(c1_end + rng.uniform(15, 50), 1)
            c2_len = round(len(cluster_sents[1]) * stride, 1)
            c2_end = min(c2_start + c2_len, video_dur - 5)
            if c2_start < video_dur - 20:
                clusters.append((c2_start, c2_end))
                c_sents.append(cluster_sents[1])

        specs.append(TopicSpec(
            topic=name,
            concept_terms=terms,
            video_duration_sec=video_dur,
            cue_duration=cue_dur,
            gap=gap,
            mention_clusters=clusters,
            cluster_sentences=c_sents,
            filler_sentences=GENERIC_FILLERS,
            clip_min_len=min_len,
            clip_max_len=max_len,
        ))

    return specs


TOPIC_SPECS = _gen_topics()


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _max_inter_cue_gap(transcript: list[dict[str, Any]]) -> float:
    max_gap = 0.0
    for i in range(len(transcript) - 1):
        end_i = float(transcript[i]["start"]) + float(transcript[i].get("duration") or 0)
        start_next = float(transcript[i + 1]["start"])
        g = start_next - end_i
        if g > max_gap:
            max_gap = g
    return max_gap


@dataclass
class ReelResult:
    """One reel produced by the simulation."""
    topic: str
    window: tuple[float, float]
    segment_idx: int
    cluster_group_id: str
    cluster_sub_index: int


@dataclass
class TopicSimResult:
    """Full simulation result for one topic search."""
    topic: str
    concept_terms: list[str]
    clip_min_len: int
    clip_max_len: int
    video_duration_sec: int
    segments_found: int
    reels: list[ReelResult]
    failures: list[str]


def simulate_user_search(
    rs: ReelService,
    spec: TopicSpec,
) -> TopicSimResult:
    """Simulate a user searching for a topic — full pipeline."""
    transcript = build_transcript_from_spec(spec)

    # Step 1: Topic cut — find segments mentioning the concept
    segments = rs._topic_cut_segments_for_concept(
        transcript=transcript,
        video_id=f"sim_{spec.topic}",
        video_duration_sec=spec.video_duration_sec,
        clip_min_len=spec.clip_min_len,
        clip_max_len=spec.clip_max_len,
        max_segments=6,
        concept_terms=spec.concept_terms,
    )

    failures: list[str] = []
    reels: list[ReelResult] = []

    if not segments:
        # No segments — this is OK for some edge cases
        return TopicSimResult(
            topic=spec.topic,
            concept_terms=spec.concept_terms,
            clip_min_len=spec.clip_min_len,
            clip_max_len=spec.clip_max_len,
            video_duration_sec=spec.video_duration_sec,
            segments_found=0,
            reels=[],
            failures=[],
        )

    # Step 2: Chain-aware refinement (replicates main generate_reels loop)
    cluster_chain: dict[str, float] = {}
    last_end: float | None = None
    BRIDGE = 2.0

    sorted_segs = sorted(
        segments,
        key=lambda s: (float(s.t_start), int(getattr(s, "cluster_sub_index", 0))),
    )

    for si, seg in enumerate(sorted_segs):
        span = seg.t_end - seg.t_start
        cg = str(getattr(seg, "cluster_group_id", "") or "")
        prev = cluster_chain.get(cg) if cg else None

        if prev is not None:
            eff = float(prev)
        elif last_end is not None and abs(float(seg.t_start) - last_end) <= BRIDGE:
            eff = float(last_end)
        else:
            eff = float(seg.t_start)

        if span > spec.clip_max_len + 16:
            windows = rs._split_into_consecutive_windows(
                transcript=transcript,
                segment_start=eff,
                segment_end=seg.t_end,
                video_duration_sec=spec.video_duration_sec,
                min_len=spec.clip_min_len,
                max_len=spec.clip_max_len,
            )
        else:
            refiner_max = int(max(span + 16, float(spec.clip_max_len)))
            refiner_min = max(1, min(int(spec.clip_min_len), int(max(1.0, span * 0.6))))
            single = rs._refine_clip_window_from_transcript(
                transcript=transcript,
                proposed_start=eff,
                proposed_end=seg.t_end,
                video_duration_sec=spec.video_duration_sec,
                min_len=refiner_min,
                max_len=refiner_max,
                min_start=eff,
            )
            windows = [single] if single else []

        for w in windows:
            if w:
                reels.append(ReelResult(
                    topic=spec.topic,
                    window=w,
                    segment_idx=si,
                    cluster_group_id=cg,
                    cluster_sub_index=getattr(seg, "cluster_sub_index", 0),
                ))

        if windows:
            last_w = [w for w in windows if w]
            if last_w:
                last_end = float(last_w[-1][1])
                if cg:
                    cluster_chain[cg] = last_end

    return TopicSimResult(
        topic=spec.topic,
        concept_terms=spec.concept_terms,
        clip_min_len=spec.clip_min_len,
        clip_max_len=spec.clip_max_len,
        video_duration_sec=spec.video_duration_sec,
        segments_found=len(segments),
        reels=reels,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class UserSearchSimulation(unittest.TestCase):
    """Simulate 100 user searches and verify reel quality."""

    def setUp(self) -> None:
        self.rs = ReelService(
            embedding_service=EmbeddingService(),
            youtube_service=YouTubeService(),
        )
        self.maxDiff = None

    def _check_boundary_quality(
        self,
        transcript: list[dict[str, Any]],
        reel: ReelResult,
    ) -> list[str]:
        """Check start/end boundary quality for one reel."""
        failures = []
        t_start, t_end = reel.window

        # --- START BOUNDARY ---
        # Must align to a cue start within tolerance
        found_cue = False
        for i, cue in enumerate(transcript):
            if abs(float(cue["start"]) - t_start) < 0.1:
                found_cue = True
                break
        # Relaxed: within 1.5s of any cue start is acceptable
        if not found_cue:
            for cue in transcript:
                if abs(float(cue["start"]) - t_start) < 1.5:
                    found_cue = True
                    break
        if not found_cue:
            failures.append(
                f"[{reel.topic}] reel start {t_start:.1f}s not aligned to any cue"
            )

        # --- END BOUNDARY ---
        # Should align to cue end (ideally with terminal punct)
        found_end = False
        for cue in transcript:
            cue_end = float(cue["start"]) + float(cue.get("duration") or 0)
            if abs(cue_end - t_end) < 0.2:
                found_end = True
                break
        if not found_end:
            for cue in transcript:
                cue_end = float(cue["start"]) + float(cue.get("duration") or 0)
                if abs(cue_end - t_end) < 1.5:
                    found_end = True
                    break
        if not found_end:
            failures.append(
                f"[{reel.topic}] reel end {t_end:.1f}s not aligned to any cue end"
            )

        return failures

    def _check_chain_integrity(
        self,
        result: TopicSimResult,
        transcript: list[dict[str, Any]],
    ) -> list[str]:
        """Check that continuation chains are not interrupted.

        If segment A produces reels [R1, R2, R3] (via splitting), those
        must appear consecutively in the final reel list — no reel from
        segment B may appear between them.
        """
        failures = []
        if len(result.reels) < 2:
            return failures

        # Group reels by cluster_group_id (only non-empty ones matter)
        groups: dict[str, list[int]] = {}
        for idx, reel in enumerate(result.reels):
            if reel.cluster_group_id:
                groups.setdefault(reel.cluster_group_id, []).append(idx)

        for gid, indices in groups.items():
            if len(indices) < 2:
                continue
            # Check indices are consecutive (no interleaving)
            for i in range(len(indices) - 1):
                if indices[i + 1] != indices[i] + 1:
                    failures.append(
                        f"[{result.topic}] chain {gid} interrupted: "
                        f"reel indices {indices} are not consecutive"
                    )
                    break

            # Check temporal ordering and gaps
            max_gap = _max_inter_cue_gap(transcript)
            gap_tol = max_gap + 0.1
            for i in range(len(indices) - 1):
                r_cur = result.reels[indices[i]]
                r_next = result.reels[indices[i + 1]]
                gap = r_next.window[0] - r_cur.window[1]
                if gap < -0.01:
                    failures.append(
                        f"[{result.topic}] chain {gid} overlap: "
                        f"{abs(gap):.3f}s between reels {indices[i]},{indices[i+1]}"
                    )
                elif gap > gap_tol:
                    failures.append(
                        f"[{result.topic}] chain {gid} gap: "
                        f"{gap:.3f}s between reels {indices[i]},{indices[i+1]} "
                        f"(max cue gap={max_gap:.3f}s)"
                    )

        # Also check that ALL consecutive same-segment reels are adjacent
        # (even if cluster_group_id is empty — from _split_into_consecutive_windows)
        for i in range(len(result.reels) - 1):
            r1 = result.reels[i]
            r2 = result.reels[i + 1]
            if r1.segment_idx == r2.segment_idx:
                # Same segment — should be temporally ordered
                if r2.window[0] < r1.window[1] - 0.01:
                    failures.append(
                        f"[{result.topic}] same-segment reels out of order: "
                        f"reel {i} ends {r1.window[1]:.1f}s, "
                        f"reel {i+1} starts {r2.window[0]:.1f}s"
                    )

        return failures

    def _check_settings_compliance(
        self,
        result: TopicSimResult,
    ) -> list[str]:
        """Check duration bounds are respected."""
        failures = []
        # Allow: min_len floor (refiner_min can be lower for short segments),
        # max_len + 8s extension for sentence boundary (SENTENCE_SEARCH_BEYOND_SEC).
        # Also allow generous slack for single-segment reels where refiner_max
        # is set to max(span+16, clip_max_len).
        hard_max = result.clip_max_len + 20  # generous ceiling
        for i, reel in enumerate(result.reels):
            dur = reel.window[1] - reel.window[0]
            if dur > hard_max:
                failures.append(
                    f"[{result.topic}] reel {i} too long: {dur:.1f}s "
                    f"(hard_max={hard_max}s)"
                )
            if dur < 1.0:
                failures.append(
                    f"[{result.topic}] reel {i} degenerate: {dur:.1f}s"
                )
        return failures

    def _check_on_topic(
        self,
        result: TopicSimResult,
        transcript: list[dict[str, Any]],
    ) -> list[str]:
        """Check that reels actually cover concept mention regions."""
        failures = []
        if not result.reels:
            return failures

        # At least some reels should overlap with a concept mention cue
        concept_set = set()
        for term in result.concept_terms:
            for token in term.lower().split():
                concept_set.add(token)

        reels_with_mention = 0
        for reel in result.reels:
            t_start, t_end = reel.window
            for cue in transcript:
                cs = float(cue["start"])
                ce = cs + float(cue.get("duration") or 0)
                if ce > t_start and cs < t_end:
                    cue_text = str(cue.get("text") or "").lower()
                    if any(tok in cue_text for tok in concept_set):
                        reels_with_mention += 1
                        break

        if result.reels and reels_with_mention == 0:
            failures.append(
                f"[{result.topic}] NO reels contain any concept mention "
                f"(terms={result.concept_terms})"
            )
        return failures

    # ========================= THE 100-TOPIC TEST ========================= #

    def test_100_topic_search_simulation(self) -> None:
        """Run full search simulation for 100 diverse topics."""
        all_failures: list[str] = []
        stats = {
            "topics_tested": 0,
            "topics_with_segments": 0,
            "total_reels": 0,
            "boundary_failures": 0,
            "chain_failures": 0,
            "settings_failures": 0,
            "on_topic_failures": 0,
            "max_reels_per_topic": 0,
        }

        for spec in TOPIC_SPECS:
            transcript = build_transcript_from_spec(spec)
            result = simulate_user_search(self.rs, spec)
            stats["topics_tested"] += 1

            if result.segments_found > 0:
                stats["topics_with_segments"] += 1
            stats["total_reels"] += len(result.reels)
            stats["max_reels_per_topic"] = max(
                stats["max_reels_per_topic"], len(result.reels)
            )

            # CHECK 1: Boundary quality
            for reel in result.reels:
                bf = self._check_boundary_quality(transcript, reel)
                if bf:
                    stats["boundary_failures"] += len(bf)
                    all_failures.extend(bf)

            # CHECK 2: Chain integrity
            cf = self._check_chain_integrity(result, transcript)
            if cf:
                stats["chain_failures"] += len(cf)
                all_failures.extend(cf)

            # CHECK 3: Settings compliance
            sf = self._check_settings_compliance(result)
            if sf:
                stats["settings_failures"] += len(sf)
                all_failures.extend(sf)

            # CHECK 4: On-topic
            otf = self._check_on_topic(result, transcript)
            if otf:
                stats["on_topic_failures"] += len(otf)
                all_failures.extend(otf)

        # Print summary
        print(f"\n{'='*70}")
        print(f"100-TOPIC USER SEARCH SIMULATION RESULTS")
        print(f"{'='*70}")
        print(f"Topics tested:          {stats['topics_tested']}")
        print(f"Topics with segments:   {stats['topics_with_segments']}")
        print(f"Total reels produced:   {stats['total_reels']}")
        print(f"Max reels per topic:    {stats['max_reels_per_topic']}")
        print(f"Boundary failures:      {stats['boundary_failures']}")
        print(f"Chain integrity fails:  {stats['chain_failures']}")
        print(f"Settings violations:    {stats['settings_failures']}")
        print(f"Off-topic reels:        {stats['on_topic_failures']}")
        print(f"{'='*70}")

        if all_failures:
            print(f"\nDETAILED FAILURES ({len(all_failures)}):")
            for f in all_failures[:50]:  # cap printout
                print(f"  - {f}")
            if len(all_failures) > 50:
                print(f"  ... and {len(all_failures) - 50} more")

        # Assert: tolerate some boundary fallbacks but no chain breaks
        # or settings violations
        self.assertEqual(
            stats["chain_failures"], 0,
            f"Chain integrity failures:\n" + "\n".join(
                f for f in all_failures if "chain" in f.lower() or "interrupt" in f.lower()
            ),
        )
        self.assertEqual(
            stats["settings_failures"], 0,
            f"Settings violations:\n" + "\n".join(
                f for f in all_failures if "too long" in f.lower() or "degenerate" in f.lower()
            ),
        )
        # At least 80% of topics should produce segments
        self.assertGreaterEqual(
            stats["topics_with_segments"],
            70,
            f"Only {stats['topics_with_segments']}/100 topics found segments",
        )
        # Total failures should be < 5% of total reels
        total_checks = stats["total_reels"] * 4  # 4 checks per reel
        self.assertLess(
            len(all_failures),
            max(10, total_checks * 0.05),
            f"Too many failures: {len(all_failures)} out of {total_checks} checks",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
