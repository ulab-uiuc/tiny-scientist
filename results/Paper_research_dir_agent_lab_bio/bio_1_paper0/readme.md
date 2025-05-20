
# Development of Antiviral Compounds Targeting H1N1

## Overview
This repository contains the research and code related to the computational development of antiviral compounds targeting the H1N1 influenza virus. Our work leverages advanced machine learning techniques, particularly a graph-based neural network architecture enhanced with reinforcement learning and transfer learning, to predict molecular binding affinities. This methodology aims to offer a robust framework for rapid adaptation to mutating viral threats and enhance the efficiency of antiviral drug discovery.

## Project Structure
- `data/`: Contains datasets used in training and testing the model, including genomic information and chemical structures.
- `models/`: Scripts and saved model weights for the graph-based neural network, reinforcement learning modules, and transfer learning components.
- `notebooks/`: Jupyter notebooks showcasing the experiments, data preprocessing, and model evaluation.
- `src/`: Source code implementing the graph neural network, reinforcement learning integration, and the transfer learning pipeline.
- `results/`: Outputs from experiments including model predictions, evaluation metrics, and ablation studies.

## Key Concepts
- **Graph-Based Neural Networks**: We represent molecules as graphs to capture spatial and sequential properties, utilizing graph neural networks to process these representations and predict binding affinities.
- **Reinforcement Learning**: Integrated to optimize the prediction process through a reward-based system, enhancing the model's ability to identify effective inhibitors.
- **Transfer Learning**: Employs pre-trained models on influenza data to adapt quickly to H1N1-specific datasets, mitigating the scarcity of directly related data.

## Experimental Setup
Our experiments use datasets sourced from databases like ChEMBL and PubChem and focus on known antiviral compounds. The setup includes specific hyperparameter configurations for training, split ratios for dataset partitioning, and the implementation environment.

## Results
- Achieved a mean square error (MSE) of 0.021 in predicting binding affinities.
- Precision and recall were measured at 88% and 82%, respectively.
- Ablation studies highlight the contribution of reinforcement and transfer learning to model performance.

## Future Work
Further efforts will focus on experimental validation of promising compounds and extending the methodology to other viral targets. The pursuit of richer H1N1-specific datasets and exploration of additional machine learning approaches are also prioritized.

## Contributing
We welcome contributions from the community to enhance this project. Please refer to the `CONTRIBUTING.md` file for guidelines on contributing to this repository.

## References
Relevant literature and references can be found in the `references.bib` file included in this repository.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

```
