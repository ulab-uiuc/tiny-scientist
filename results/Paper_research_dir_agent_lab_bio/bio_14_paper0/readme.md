
# Subtle Manipulation of Emotional Responses Using Advanced BCIs

## Overview
This repository contains the implementation and findings from our research on the subtle manipulation of emotional responses using advanced Brain-Computer Interfaces (BCIs). We investigated the use of connectivity-based EEG neurofeedback with simultaneous fMRI to modulate emotional states and impact decision-making processes undetectably. Our study holds significant therapeutic potential for emotional and cognitive disorder management, offering non-invasive alternatives for emotional regulation.

## Contents

- **Paper**: Detailed research report in LaTeX format.
- **Code**: Scripts for preprocessing, modeling, and analyzing EEG and fMRI data.
- **Data**: Link to the MAHNOB-HCI dataset used for training and evaluation (due to licensing, the data itself must be obtained independently).
- **Models**: Implementation of the hypercomplex neural network architecture (PHemoNet) used for the study.

## Key Contributions

- **Dual-Modality Approach**: Integration of high temporal resolution EEG and superior spatial fMRI for real-time emotion modulation.
- **Connectivity-Based Neurofeedback**: Utilization of BCIs to interpret neural activity and provide feedback, focusing on coherence between multiple brain regions.
- **Hypercomplex Neural Networks**: Advanced signal analysis through parameterized hypercomplex multiplications for accurate emotional state predictions.
- **Experimental Validation**: Controlled experiments demonstrating significant changes in EEG frontal asymmetry and BOLD signals, validated by statistical analyses.

## Getting Started

### Prerequisites

- Python 3.x
- Required packages: numpy, scipy, sklearn, tensorflow, keras, and other dependencies listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/emotional-responses-bci.git
   ```
2. Navigate to the project directory:
   ```bash
   cd emotional-responses-bci
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preparation**: Obtain the MAHNOB-HCI dataset and organize it as specified in `data/README.md`.
2. **Run Preprocessing**: Execute the preprocessing scripts for EEG and fMRI data:
   ```bash
   python preprocess.py
   ```
3. **Train the Model**: Train the hypercomplex neural network:
   ```bash
   python train_model.py
   ```
4. **Evaluate the Model**: Run evaluations and export results:
   ```bash
   python evaluate.py
   ```

## Results

- **EEG Frontal Asymmetry**: Increased from 0.2 to 0.35 post-intervention.
- **BOLD Signal in Insula**: Increased by 15%, revealing enhanced emotion regulation capabilities.
- **Connectivity**: Boosted coherence between the prefrontal cortex and insula by 25%.

## Discussion

Our findings support the hypothesis that connectivity-based EEG neurofeedback can effectively manipulate emotional states, impacting decision-making. This approach opens new avenues for emotion regulation strategies with therapeutic implications for emotional and cognitive disorders.

## Future Work

- **Demographic Diversity**: Including a more diverse participant sample for greater generalizability.
- **Real-Time Data Collection**: Enhancing real-world applicability.
- **Exploration of Other Emotions**: Expanding the emotional states and cognitive processes targeted by neurofeedback interventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or collaboration interests, please contact [author's email].

```
```