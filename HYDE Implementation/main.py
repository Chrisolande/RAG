import os
from dotenv import load_dotenv
from hyde import (connect_to_pinecone, create_pinecone_index, insert_documents_to_pinecone,
                search_pinecone, initialize_llm, hyde_search)

# Load environment variables
load_dotenv()

# Test corpus of documents
corpus = [
    """1. **Penicillin Discovery**: In 1928, Alexander Fleming discovered penicillin, the first widely used antibiotic, when he noticed that a mold growing on a petri dish in his lab inhibited bacterial growth. 
This accidental observation led to the isolation of penicillin, which transformed medicine by providing an effective treatment for bacterial infections like pneumonia and sepsis, previously often fatal. 
By the 1940s, mass production of penicillin saved countless lives during World War II, marking the beginning of the antibiotic era, though overuse later contributed to antimicrobial resistance.""",

    """2. **General Relativity**: Albert Einstein's theory of general relativity, published in 1915, redefined gravity as the curvature of spacetime caused by mass and energy, overturning Newton's classical model. 
This theory explained phenomena like the precession of Mercury's orbit and predicted effects such as gravitational lensing, later confirmed by observations. 
Its practical applications include the precise functioning of GPS satellites, which account for time dilation caused by Earth's gravitational field, revolutionizing navigation and telecommunications.""",

    """3. **DNA Double Helix**: In 1953, James Watson and Francis Crick, building on Rosalind Franklin's X-ray diffraction data, elucidated the double-helix structure of DNA at Cambridge University. 
This discovery revealed how genetic information is stored and replicated, laying the groundwork for molecular biology. 
It enabled advancements like the Human Genome Project, genetic engineering, and personalized medicine, where treatments are tailored to an individual's genetic profile, transforming healthcare and biotechnology.""",

    """4. **Steam Engine**: James Watt's improvements to the steam engine in the 1760s and 1770s, particularly the addition of a separate condenser, vastly increased efficiency, making it a cornerstone of the Industrial Revolution. 
By powering machinery, locomotives, and ships, the steam engine drove mass production and global trade, reshaping economies and urban landscapes. 
Its legacy persists in modern energy systems, though it also contributed to early environmental challenges like coal-based pollution.""",

    """5. **Quantum Computing**: Emerging in the 1980s through work by physicists like Richard Feynman and David Deutsch, quantum computing uses quantum mechanics principles, such as superposition and entanglement, to process information exponentially faster than classical computers for specific tasks. 
Unlike binary bits, quantum bits (qubits) can represent multiple states simultaneously, promising breakthroughs in cryptography, drug discovery, and optimization problems. 
Companies like IBM and Google are advancing practical quantum computers, though challenges like error rates remain.""",

    """6. **mRNA Vaccines**: The development of mRNA vaccines, pioneered by researchers like Katalin Karikó and Drew Weissman in the early 2000s, enabled rapid response to the COVID-19 pandemic in 2020. 
These vaccines deliver mRNA that instructs cells to produce a viral protein, triggering an immune response without using live virus. 
Their adaptability and speed of production have revolutionized vaccine development, offering potential applications for diseases like cancer and HIV, though distribution challenges persist in low-resource settings.""",

    """7. **Exoplanet Discovery**: The first confirmed exoplanet, 51 Pegasi b, was discovered in 1995 by Michel Mayor and Didier Queloz using the radial velocity method, which detects stellar wobbles caused by orbiting planets. 
This finding expanded our understanding of planetary systems, revealing diverse worlds, including “hot Jupiters” and potentially habitable exoplanets. 
Over 5,000 exoplanets have since been identified, driving research into astrobiology and the search for extraterrestrial life via telescopes like Kepler and TESS.""",

    """8. **Polymerase Chain Reaction (PCR)**: Invented by Kary Mullis in 1983, the polymerase chain reaction (PCR) is a technique that amplifies small DNA segments, enabling detailed genetic analysis. 
By cycling through temperature changes to replicate DNA, PCR became essential for genetic testing, forensic science, and diagnosing diseases like COVID-19. 
Its simplicity and precision have made it a cornerstone of biotechnology, though it requires careful control to avoid contamination errors.""",

    """9. **World Wide Web**: In 1989, Tim Berners-Lee proposed the World Wide Web at CERN, creating a system to share information via hyperlinked documents over the internet. 
By developing protocols like HTTP and HTML, he enabled global access to interconnected data, transforming communication, education, and commerce. 
The web's open architecture spurred innovations like social media and e-commerce, though it also raised challenges like misinformation and privacy concerns.""",

    """10. **Gravitational Waves**: In 2015, the Laser Interferometer Gravitational-Wave Observatory (LIGO) detected gravitational waves, ripples in spacetime caused by the merger of two black holes, confirming a prediction of Einstein's general relativity. 
This discovery opened a new field of astronomy, allowing scientists to study cosmic events like neutron star collisions, which produce heavy elements like gold. 
LIGO's precision measurements continue to deepen our understanding of the universe's most extreme phenomena.""",

    """11. **Lithium-Ion Batteries**: Developed in the 1980s by John Goodenough, Akira Yoshino, and others, lithium-ion batteries store energy through the movement of lithium ions, offering high efficiency and rechargeability. 
Commercialized by Sony in 1991, they power smartphones, laptops, and electric vehicles, driving the shift to renewable energy. 
Their lightweight design and longevity have transformed technology, though recycling and ethical sourcing of materials like cobalt remain challenges.""",

    """12. **Higgs Boson**: In 2012, the Large Hadron Collider at CERN confirmed the existence of the Higgs boson, a particle proposed by Peter Higgs and others in the 1960s to explain how particles acquire mass via the Higgs field. 
This discovery completed the Standard Model of particle physics, providing insights into the universe's fundamental structure. 
It has spurred research into new physics, though the particle's fleeting nature requires massive accelerators to study.""",

    """13. **Telescope Invention**: In 1608, Hans Lippershey patented the telescope, a device that magnifies distant objects using lenses. 
Galileo Galilei's refinements enabled observations of Jupiter's moons and Saturn's rings, challenging geocentric models and supporting the Copernican revolution. 
Telescopes remain critical for astronomy, with modern versions like the James Webb Space Telescope probing the universe's earliest galaxies, advancing our cosmic knowledge.""",

    """14. **Haber-Bosch Process**: Developed by Fritz Haber and Carl Bosch in the early 20th century, the Haber-Bosch process synthesizes ammonia from nitrogen and hydrogen, enabling large-scale fertilizer production. 
This innovation boosted agricultural yields, supporting global population growth by making farming more efficient. 
However, its energy-intensive nature and environmental impacts, like nitrogen runoff, have prompted research into sustainable alternatives.""",

    """15. **Superconductivity**: Discovered by Heike Kamerlingh Onnes in 1911, superconductivity occurs when certain materials, cooled to extremely low temperatures, conduct electricity with zero resistance. 
This phenomenon enables powerful electromagnets used in MRI machines and high-speed maglev trains. 
Ongoing research into high-temperature superconductors aims to make the technology more practical, potentially revolutionizing energy transmission and storage."""
]


def main():
    # Connect to Pinecone
    connect_to_pinecone()
    llm = initialize_llm()
    # Create or connect to index with 768 dimensions
    index = create_pinecone_index(dimension=768)
    
    # Insert documents
    insert_documents_to_pinecone(index, corpus)
    

    query = "What is the significance of the discovery of gravitational waves by LIGO, and how has it impacted our understanding of the universe?"
    
    print("\n========== TRADITIONAL SEARCH ==========\n")
    # Perform traditional search
    traditional_results = search_pinecone(index, query)
    
    print("\nTraditional Search Results:")
    for doc, score in traditional_results:
        print(f"Score: {score:.4f} - {doc}")
    
    print("\n\n========== HYDE SEARCH ==========\n")
    # Perform HyDE search
    hyde_results = hyde_search(index, query, llm, use_hyde=True)
    
    print("\nHyDE Search Results:")
    for doc, score in hyde_results:
        print(f"Score: {score:.4f} - {doc}")
        
    # Additional example query for comparison
    print("\n\n========== ANOTHER EXAMPLE ==========")
    query2 = "What is the Haber-Bosch process, and what are its benefits and environmental challenges?"
    
    print("\n--- Traditional Search ---")
    trad_results2 = search_pinecone(index, query2)
    for doc, score in trad_results2:
        print(f"Score: {score:.4f} - {doc}")
        
    print("\n--- HyDE Search ---")
    hyde_results2 = hyde_search(index, query2, llm, use_hyde=True)
    for doc, score in hyde_results2:
        print(f"Score: {score:.4f} - {doc}")

if __name__ == "__main__":
    main()
    print("\nComparison complete! HyDE generates hypothetical documents that better match the relevant content.")
    print("This often leads to improved retrieval performance, especially for complex queries.")

