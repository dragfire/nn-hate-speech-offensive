# Detect offensive and hateful speech on social media
Neural net to detect hate-speech and offensive language

***WARNING: The data, lexicons, and notebooks all contain content that is racist, sexist, homophobic, and offensive in many other ways.***

##  How to train & run the classifier:
```
python train.py \
   --batch_size 64 \
   --rnn_layers 5 \
   --learning_rate 0.001 \
   --epochs 25 \
   --embedding_dim 64 \
   --hidden_size 128 \
   --dropout_prob 0.5 \
   --device mps

python classifier.py --text "All tilapias should be burnt alive"
python classifier.py --text "All tilapias should be removed from extistence from this planet. They are useless "
python classifier.py --text "they are useless"
``

or 

```
chmod +x train.sh
./train.sh
```

## Approach 1:
- Uses LSTM with dropouts and layer normalization.

## Approach 2: (TODO)
- Use transformers (encoder only)

This project utilizes a set of hand annotated datasets provided by [@crowdflower](https://data.world/crowdflower).

## References:
~~~
@inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
~~~
