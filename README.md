# Quantum Machine Learning for Stock Market Prediction
This project explores the application of Quantum Machine Learning (QML) in the field of stock market prediction. Leveraging the unique advantages of quantum computing, we have developed and implemented a Quantum Long Short-Term Memory (QLSTM) model. This model is benchmarked against various approaches including Quantum Recurrent Neural Networks (QRNN), classical LSTM, and a non-machine learning classical baseline.

## Key Features
- **Quantum LSTM Implementation:** A novel approach utilizing the principles of quantum computing for enhanced predictive analytics in stock market data.
- **Comparative Analysis:** In-depth comparison of QLSTM with QRNN, classical LSTM, and a classical baseline approach.
- **Diverse Architectures:** Implementation of different architectures within our QLSTM to optimize performance.
- **Multi-Faceted Data Inputs:** Utilization of four key data inputs for analysis:
  - **Closing Price:** Indicates the final price at which a stock is traded on a given trading day, reflecting the market's valuation of a company.
  - **Volume:** Represents the total number of shares traded in a day, which helps in understanding the market's activity and liquidity.
  - **Percentage Change:** Provides insights into the stock's short-term performance and market trends.
  - **Technical Indicator:** A mathematical calculation based on historical price, volume, or open interest information that aims to forecast financial market directions.
- **Evaluation on PennyLane and IBM-Q:** Testing and validation of models on PennyLane simulators and real quantum hardware provided by IBM-Q.

## Resources
Here you can find our [Wiki](https://gitlab.lrz.de/mobile-ifi/qcp/23ws/quantum_finance_basf/-/wikis/Papers) with some related work on the topic.

## Contributors
- Alexander Kovacs @akovacs
- Florian Eckstaller @00000000013586DE
- Leonard Niessen @00000000013540DD
- Mohamad Hgog @mhgog

## Acknowledgment
The QML project is being conducted at Ludwig Maximilian University of Munich (LMU Munich) as part of the Quantum Computing (QC) Optimization Challenge hosted by the Quantum Applications and Research Laboratory (QAR-Lab). This project is in collaboration with the industry partner BASF.
