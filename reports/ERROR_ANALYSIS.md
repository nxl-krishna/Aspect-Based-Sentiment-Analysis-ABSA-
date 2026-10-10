## Error analysis: DeBERTa-v3 aspect sentiment model

Evaluated on the official SemEval-2014 test sets (1758 aspects). Overall accuracy 85.9%; 248 errors.

![Confusion matrices](figures/confusion_matrices.png)

### Accuracy by sentence type

| Sentence type | Aspects | Accuracy |
|---|---|---|
| multi-aspect, same sentiment | 883 | 89.9% |
| single aspect | 544 | 86.6% |
| multi-aspect, mixed sentiment | 331 | 74.0% |

### Most common confusions

| True -> Predicted | Count | Share of errors |
|---|---|---|
| neutral -> positive | 100 | 40% |
| neutral -> negative | 69 | 28% |
| positive -> neutral | 29 | 12% |
| negative -> neutral | 20 | 8% |
| negative -> positive | 15 | 6% |
| positive -> negative | 15 | 6% |

### Most confident mistakes

| Domain | Sentence | Aspect | True | Predicted | Confidence |
|---|---|---|---|---|---|
| Restaurants | The sauce is excellent (very fresh) with dabs of real mozzarella. | dabs of real mozzarella | neutral | positive | 99.9% |
| Restaurants | Thick fries, meaty chili and stuffed baked potatoes round out a menu that includes a cool, ultra-thick chocolate Frosty. | meaty chili | neutral | positive | 99.9% |
| Restaurants | They are served on Focacchia bread and are to die for. | served | neutral | positive | 99.9% |
| Restaurants | The boutique selection of wines covers a wide variety without being imposeing. | boutique selection of wines | neutral | positive | 99.9% |
| Restaurants | Creamy appetizers--taramasalata, eggplant salad, and Greek yogurt (with cuccumber, dill, and garlic) taste excellent when on warm pitas. | warm pitas | neutral | positive | 99.9% |
| Restaurants | They are served on Focacchia bread and are to die for. | Focacchia bread | neutral | positive | 99.9% |
| Laptops | I needed a laptop with big storage, a nice screen and fast so I can photoshop without any problem. | screen | neutral | positive | 99.9% |
| Restaurants | Thick fries, meaty chili and stuffed baked potatoes round out a menu that includes a cool, ultra-thick chocolate Frosty. | stuffed baked potatoes | neutral | positive | 99.9% |
| Laptops | With the softwares supporting the use of other OS makes it much better. | softwares | neutral | positive | 99.9% |
| Restaurants | good place to hang out during the day after shopping or to grab a simple soup or classic french dish over a glass of wine. | french dish | neutral | positive | 99.9% |
| Laptops | I wanted a computer that was quite, fast, and that had overall great performance. | performance | neutral | positive | 99.9% |
| Restaurants | Thick fries, meaty chili and stuffed baked potatoes round out a menu that includes a cool, ultra-thick chocolate Frosty. | fries | neutral | positive | 99.9% |
