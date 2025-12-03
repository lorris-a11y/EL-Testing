# EL-Test4NER
This repository stores our experimental codes and results.

## Dataset
Our dataset is based on the BBC News dataset, which is an unlabeled dataset from Yu et al. [1] and stored in the `txt2json` folder in JSON format.

The original format of this dataset is txt. We converted it to JSON format for our subsequent experiments. You can run the `txt2json.py` script in the `txt2json` folder to perform this conversion.
The `txt2json` folder also contains the converted BBC News dataset files in JSON format. 



## Source Code
We've tested our code on Windows 11 with Python 3.8.

### Requirements
- torch == 2.4.1
- transformers == 4.46.0
- spacy
- sentence-transformers == 3.2.1


### Components

- **AwsUtils & AzureUtils**: Provide unified interfaces for AWS Comprehend and Azure Cognitive Services that mimic Flair's SequenceTagger
- **grammarRefiner**: Implements subject-verb agreement consistency checking and correction
- **external**: Singular/plural transformation utilities (based on [[external repository](https://github.com/test4mt/BDTD)])
- **repair**: Experimental code for NER system repair and improvement
- **rules**: Core mutation operators for different testing strategies:
  - Rule 1: PER mutation (entity swapping within coordinated structures)
  - Rule 2: CES mutation (entity swapping between sentence pairs)
  - Rule 3: KPS mutation (knowledge graph based entity replacement)
- **testScriptsOrig**: Individual test scripts for each rule and model.

### Model Setup

1. Place your NER model files in the `models/` directory:
   - `models/conll-large.bin` (Flair CoNLL model)
   - `models/ontonotes-large.bin` (Flair OntoNotes model)

2. For cloud services, configure your API credentials:
   - AWS: Set up AWS credentials for Comprehend service
   - Azure: Configure Azure Cognitive Services credentials

## Additional Configuration

Before running the code, please perform the following critical setup steps based on your local environment.

### 1\. Proxy Setup

To ensure a stable connection to external APIs like Wikidata, the code is configured to use a network session. You must replace the proxy placeholder with your own proxy server address.

  * Search for `your_proxy_here` within the codebase.
  * Replace it with your proxy URL.

**Example:**

```python
# Before
proxies = {
   'https': 'your_proxy_here',
}

# After
proxies = {
   'http': 'http://user:password@127.0.0.1:8080',
   'https': 'http://user:password@127.0.0.1:8080'
}
```

### 2\. File Paths

The paths for model files, input data, and output files placeholders need to be replaced with real path. You will need to locate these path variables in the code and update them to match the directory structure on your local.

  * Search for `your_test_file_here` within the codebase.
  * Replace it with the exact input data path.

  * Search for `your_mutated_file_path` within the codebase.
  * Replace it with the exact output file path.

  * Search for `your_suspicious_file_path` within the codebase.
  * Replace it with the exact output file path.


### Dataset Conversion

Convert BBC News dataset from txt to JSON format:

```bash
cd txt2json
python txt2json.py
```



### Run
Use the unified test runner to execute different mutation rules with various NER models:

```bash
# Run Rule1 (coordination mutation) with CoNLL model
python test_runner.py rule1 conll

# Run Rule2 (cross-sentence mutation) with OntoNotes model
python test_runner.py rule2 ontonotes

# Run Rule3 (entity linking mutation) with AWS model
python test_runner.py rule3 aws

# Enable verbose output
python test_runner.py rule1 azure --verbose
```
Use the unified repair runner to execute repair after mutations:
```bash
python test_repair_unified.py 
```
### Available Options

**Rules:**

  * `rule1`: Parallel Entity Reduction
  * `rule2`: Cross-sentence entity Swapping
  * `rule3`: Knowledge graph-enhanced Pronoun Substitution

**Models:**

  * `conll`: Flair CoNLL03 
  * `ontonotes`: Flair OntoNotes 
  * `aws`: AWS Comprehend NER service
  * `azure`: Azure Cognitive Services NER

**Note**: The `--verbose` flag enables detailed logging to show all outputs.

## Output

Test results are saved in respective directories:
- **TR-Mode1/**: Results for CoNLL model tests
- **TR-Mode2/**: Results for OntoNotes model tests  
- **TR-AWS/**: Results for AWS NER tests
- **TR-Azure/**: Results for Azure NER tests

Each test generates:
- `mutated_results*.json`: Successfully generated mutations
- `suspicious_sentences*.json`: Detected inconsistencies and suspicious cases
- Log files with detailed execution information


## Results
The `results` folder contains the full results of our experiment. Both mutation and repair.


## References
[1] Boxi Yu, Yiyan Hu, Qiuyang Mang, Wenhan Hu, and Pinjia He. 2023. Automated Testing and Improvement of Named Entity Recognition Systems. In Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2023). ACM, 883–894. https://doi.org/10.1145/3611643.3616295





