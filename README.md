# ds4finance

A collection of data science tools for finance.

## Installation

Install the package from PyPI using pip:

```python
pip install ds4finance
```

## Usage 

To use the package, import it using the alias dsf:

import ds4finance as dsf

### Example: Computing Standard Deviation

```python
import pandas as pd
import ds4finance as dsf

# Load or create a DataFrame containing financial data
data = pd.read_csv("your_data.csv")

# Compute the annualized standard deviation for the data
std_dev = dsf.compute_std_dev(data)
```

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for more information.

## Contributing

We welcome contributions to the project. If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes, and ensure that the code follows PEP 8 guidelines.
4. Submit a pull request to the main repository.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/LuisSousaSilva/ds4finance/issues) on the GitHub repository.

## Authors

- Luis Silva - luis_paulo_silva@hotmail.com
