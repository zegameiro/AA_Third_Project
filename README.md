# Advanced Algorithms Third Project

## Most Frequent and Less Frequent Words

The objective of this project is to analyse different strategies that identify frequent words in text files, like books, using three different methods: exact counters, approximate counters (with a decreasing probability counter of 1 / sqrt(2)^k) and to evaluate the quality of estimates regarding the exact counts.

### Folder Structure

```bash
.
├── data
│   ├── don_quixote_en.txt
│   ├── don_quixote_ger.txt
│   ├── don_quixote_hun.txt
│   └── don_quixote_sp.txt
├── docs
│   ├── AA_2425_Trab_3.pdf
│   ├── AA_Third_Project_Report.pdf
│   ├── Assignments_TP1.pdf
│   └── Assignments_TP2.pdf
├── images
│   ├── don_quixote_english_approximate_counter_absolute_errors.png
│   ├── don_quixote_english_approximate_counter_relative_errors.png
│   ├── don_quixote_english_k_100_space_saving_counter_absolute_errors.png
│   ├── don_quixote_english_k_100_space_saving_counter_relative_errors.png
│   ├── don_quixote_english_k_20_space_saving_counter_absolute_errors.png
│   ├── don_quixote_english_k_20_space_saving_counter_relative_errors.png
│   ├── don_quixote_english_k_35_space_saving_counter_absolute_errors.png
│   ├── don_quixote_english_k_35_space_saving_counter_relative_errors.png
│   ├── don_quixote_german_approximate_counter_absolute_errors.png
│   ├── don_quixote_german_approximate_counter_relative_errors.png
│   ├── don_quixote_german_k_100_space_saving_counter_absolute_errors.png
│   ├── don_quixote_german_k_100_space_saving_counter_relative_errors.png
│   ├── don_quixote_german_k_20_space_saving_counter_absolute_errors.png
│   ├── don_quixote_german_k_20_space_saving_counter_relative_errors.png
│   ├── don_quixote_german_k_35_space_saving_counter_absolute_errors.png
│   ├── don_quixote_german_k_35_space_saving_counter_relative_errors.png
│   ├── don_quixote_hungarian_approximate_counter_absolute_errors.png
│   ├── don_quixote_hungarian_approximate_counter_relative_errors.png
│   ├── don_quixote_hungarian_k_100_space_saving_counter_absolute_errors.png
│   ├── don_quixote_hungarian_k_100_space_saving_counter_relative_errors.png
│   ├── don_quixote_hungarian_k_20_space_saving_counter_absolute_errors.png
│   ├── don_quixote_hungarian_k_20_space_saving_counter_relative_errors.png
│   ├── don_quixote_hungarian_k_35_space_saving_counter_absolute_errors.png
│   ├── don_quixote_hungarian_k_35_space_saving_counter_relative_errors.png
│   ├── don_quixote_spanish_approximate_counter_absolute_errors.png
│   ├── don_quixote_spanish_approximate_counter_relative_errors.png
│   ├── don_quixote_spanish_k_100_space_saving_counter_absolute_errors.png
│   ├── don_quixote_spanish_k_100_space_saving_counter_relative_errors.png
│   ├── don_quixote_spanish_k_20_space_saving_counter_absolute_errors.png
│   ├── don_quixote_spanish_k_20_space_saving_counter_relative_errors.png
│   ├── don_quixote_spanish_k_35_space_saving_counter_absolute_errors.png
│   └── don_quixote_spanish_k_35_space_saving_counter_relative_errors.png
├── LICENSE
├── README.md
├── requirements.txt
├── results
│   ├── don_quixote_english_results.txt
│   ├── don_quixote_german_results.txt
│   ├── don_quixote_hungarian_results.txt
│   └── don_quixote_spanish_results.txt
└── src
    └── main.py

6 directories, 48 files
```

### How to execute

1. Open a terminal in the the root directory of the project and create a virtual environment using this command:

```bash
python3 -m venv venv
```

2. Activate the created virtual environment:

```bash
source venv/bin/activate
```

3. Install the necessary requirements:

```bash
pip install -r requirements.txt
```

4. Navigate to the src directory

```bash
cd src/
```

5. Run the __main.py__ file with the following options:

```bash
python3 main.py --file <path to a text file inside the data folder>
```

__Note:__ The `--file` argument is mandatory and it should contain the correct path to the text file. The text files used should only be the ones inside the data folder since each has a different language and the program uses different stopword lists for each language. After the execution is complete, the results will be stored in the results folder and the plots will be stored in the images folder.

### Authors

José Gameiro 108840

