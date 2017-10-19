# Mixuture of Probabilistic Partial Canonical Correlation Analysis
Code for our paper *Causal Patterns: Extraction of Multiple Causal Relationships by Mixture of Probabilistic Partial Canonical Correlation Analysis*.

## Publication
Hiroki Mori, Keisuke Kawano and Hiroki Yokoyama. "*Causal Patterns: Extraction of Multiple Causal Relationships by Mixture of Probabilistic Partial Canonical Correlation Analysis*" Proceedings of the 2017 IEEE International Conference on Data Science and Advanced Analytics. pp.744-754

## Dependencies
- for mppcca.py
  - Python == 3.6
  - numpy== 1.13.1
  - scipy == 0.19.1
- for the sample programs in example directory, some additional libraries are required.
  - matplotlib
  - seaborn
  - scikit-learn

## Setup
```sh
$ git clone https://github.com/kskkwn/mppcca.git
$ cd mppcca
$ conda env create --file env.yaml # if you use anaconda else install above dependencies manually.
$ source activate mppcca
```

## Examples

```sh
$ python example/toy_scatter_data/toy_scatter_data.py 
$ python example/time_series_exp/time_series_exp.py
```

### "Extraction of Multiple Causal Relationships" from your own data
```sh
$ python mppcca_from_csv.py -i example/data.csv -k 3 -d 1 -e 1 -o temp.csv
$ python mppcca_from_csv.py -i ${input_file} -k ${nb_clusters} -d ${delay_time} -e ${embedding_time} -o ${output_file}
```


