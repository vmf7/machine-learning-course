{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I found an interesting data set on all Ted Talks uploaded to Ted.com up\n",
    "to and including September 21st, 2017. This is the link for the dataset:\n",
    "<https://www.kaggle.com/rounakbanik/ted-talks/data>. This dataset\n",
    "includes information on number of views and comments. It also includes a\n",
    "description of each talk, along with corresponding titles and speakers.\n",
    "I found this data set to be interesting because by applying analytical\n",
    "methods, it is possible to reverse-engineer the \\\"optimal\\\" Ted Talk\n",
    "based on its content. For example, I am curious whether Ted Talks with\n",
    "different themes have different optimal lengths. This dataset would shed\n",
    "light as to whether there is a different ideal length of a Ted Talk in\n",
    "the *Funny* category versus the *Inspiring* Category (in order to\n",
    "maximize views and comments). Additionally, I wonder if certain words or\n",
    "phrases in titles tend to attract more views and responses.\n",
    "\n",
    "There are no missing values in this data set, so data cleansing is not\n",
    "necessary. Below is a plot the number of views for each Ted Talk. I\n",
    "chose to plot this as a stacked column graph because I was curious if\n",
    "visualizing the data would indicate whether there are any trends in\n",
    "number of views. The plot is below.\n",
    "\n",
    "<img src = \"stackedcolumngraph.png\">\n",
    "\n",
    "\n",
    "I find it interesting very few Ted Talks receive over 2,000,000 views. I\n",
    "wonder if there is any commonality regarding style, content, and\n",
    "duration of speech among these Ted Talks that did exceptionally well. To\n",
    "better view the frequency of number of views across all Ted Talks, a\n",
    "histogram would be helpful. The histogram of this data is below.\n",
    "\n",
    "\n",
    "<img src = \"histogram.png\">\n",
    "\n",
    "\n",
    "From exploring the data, I am able to conclude that the Ted\n",
    "Talks that receive over 2,000,000 views are indeed outliers. There must\n",
    "be a underlying trend that explains the distribution of views. This is\n",
    "likely categorical, including titles and tags in the posted videos. The\n",
    "next step in exploring this data would be to use a classification\n",
    "algorithm to sort the data based off of both content categories (like\n",
    "funny inspiring, etc.), duration in order to determine if further\n",
    "relationships exist between these categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
