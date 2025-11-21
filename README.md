# SettleUp

**A smart debt settlement visualization tool that simplifies complex transaction networks using graph-based optimization.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

SettleUp transforms messy, interconnected payment transactions into a minimal set of optimized settlements. Perfect for splitting bills among groups, settling shared expenses, or simplifying financial relationships.

### Key Features

- **Transaction Reduction**: Automatically minimizes the number of settlements needed
- **Visual Network Graphs**: Beautiful curved-edge visualizations showing before/after optimization
- **Multiple Input Methods**: Quick examples, manual entry, or CSV upload
- **Interactive Analysis**: Step-by-step breakdown of the optimization process
- **Export Capabilities**: Download both original and optimized transaction sets

### How It Works

SettleUp uses a greedy algorithm approach:

1. **Calculate Net Balances**: Determines who owes money and who should receive money
2. **Sort by Amount**: Orders debtors and creditors by transaction size
3. **Match Optimally**: Pairs largest debts with largest credits progressively
4. **Minimize Transactions**: Reduces N complex transactions to the theoretical minimum

**Example**: 8 interconnected transactions → 4 optimized settlements (50% reduction)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/settleup.git
   cd settleup
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   
   Open your browser and navigate to `http://localhost:8501`

---

## Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
networkx>=3.1
plotly>=5.14.0
```

Create a `requirements.txt` file with the above contents.

---

## Usage

### Quick Start

1. Launch the application
2. Select "Quick Example" in the sidebar
3. Click "Load & Analyze"
4. View the transaction reduction visualization

### Manual Entry

1. Select "Manual Entry" in the sidebar
2. Specify the number of transactions
3. Enter payer, payee, and amount for each transaction
4. Click "Load & Analyze"

### CSV Upload

#### Format Requirements

Your CSV must contain three columns (case-insensitive):
- `payer`: Person making the payment
- `payee`: Person receiving the payment  
- `amount`: Transaction amount (numeric)

#### Example CSV

```csv
payer,payee,amount
Alice,Bob,100
Bob,Charlie,150
Charlie,Alice,50
David,Alice,75
```

#### Upload Process

1. Select "Upload CSV" in the sidebar
2. Either upload a file or paste CSV data directly
3. Click "Load & Analyze"

---

## Understanding the Visualizations

### Original Transactions Network
- **Gray edges**: Individual payment transactions
- **Thinner lines**: Standard transaction flow
- Shows the complexity of the original transaction web

### Optimized Settlement Network
- **Green edges**: Minimized settlement transactions
- **Thicker lines**: Optimized payment paths
- Demonstrates the simplified transaction structure

### Graph Elements
- **Blue nodes**: People/entities involved in transactions
- **Node size**: Proportional to number of connections
- **Edge labels**: Transaction amounts in currency
- **Curved edges**: Improved visibility and aesthetics

---

## Algorithm Details

### Greedy Settlement Algorithm

```
1. Calculate net balance for each person:
   - net_balance = total_received - total_paid

2. Separate into two groups:
   - Debtors: net_balance < 0 (owe money)
   - Creditors: net_balance > 0 (should receive money)

3. Sort both groups by absolute amount (descending)

4. While debtors and creditors exist:
   - Match largest debtor with largest creditor
   - Transfer minimum of (debt_amount, credit_amount)
   - Update balances and remove settled parties

5. Return list of optimized settlements
```

### Complexity
- **Time Complexity**: O(n log n) where n is the number of people
- **Space Complexity**: O(n)
- **Optimality**: Produces minimal number of transactions for debt resolution

---

## Output Files

### Generated CSV Files

#### `original_transactions.csv`
Contains the input transaction data as loaded

#### `optimized_settlements.csv`
Contains the minimized settlement plan with columns:
- `from`: Payer in the settlement
- `to`: Payee in the settlement
- `amount`: Settlement amount

---

## Use Cases

### Personal Finance
- **Trip expenses**: Settle shared costs after group trips
- **Roommate bills**: Simplify rent and utility splits
- **Event costs**: Resolve shared expenses for parties or events

### Educational
- **Algorithm visualization**: Demonstrate graph theory and optimization
- **Teaching tool**: Explain greedy algorithms and network reduction
- **Data structures**: Show practical applications of graphs


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Graph algorithms powered by [NetworkX](https://networkx.org/)
- Visualizations created using [Plotly](https://plotly.com/)
- Inspiration : Faced a similar problem on campus


---

**Made with ❤️ for simplifying financial settlements**
