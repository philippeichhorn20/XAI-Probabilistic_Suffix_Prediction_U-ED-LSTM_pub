***In the following is a list of visualizations that should be supported through notebooks.
The core functionality shoud be implemented in classes, the notebooks themselves should be lean.

When selecting a categorical value as the attribution target, always use the predicted value as the attribution target.
***

## Attribution scores

1. Single Suffix Level:

- Enter a single case (via case name or id) + prefix length and output. Also provide the attribution target:
    -> Give a full attribution matrix for every step of the suffix prediction.
    -> try to give very nice visualization using an interface of your choice that lets you move through the suffix steps

2. Full dataset level:

-   Allow for either full load (calculate all attributions for all case, prefix lengthss in the dataset)
 or a test load, that just checks that it wirks with a subset.

-> Average feature attribution (for activity, resources and the 2 temporal components) w.r.t all inpiut features (4 views)
-> Average feature attribtion for the 5 most common categorical values for activity and resource w.r.t all inpiut features (10 views)

3. Decoder split analysis (over full dataset):
- Encoder vs decoder attribution by feature
- Encoder vs decoder attribution by prefix length

4. Variant index visualizer:
- Lets me enter a variant index and shows, how all of the cases of that variant index look like in terms of activites

