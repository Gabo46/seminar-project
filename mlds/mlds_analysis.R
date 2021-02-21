# MLDS analysis script

# change to your working directory. Change it to wherever you 
# have your files in your system 
setwd("C:/Users/Gabriel/seminar-project")

# First, you need to install the package. 
# you only need to do this once, you can comment 
# this line after done it for the first time.
#install.packages('MLDS')


# load package 
library(MLDS)

########################################################
## Reading and preparing data
# read all the CSV files you generated
  
# example data with quadruples
#  condition RED
#d1 <- read.csv('design_GA_quadruples_0_results_red.csv')
#d2 <- read.csv('design_GA_quadruples_1_results_red.csv')
#d3 <- read.csv('design_GA_quadruples_2_results_red.csv')


# condition BLUE 
#d1 <- read.csv('design_GA_quadruples_0_results_blue.csv')
#d2 <- read.csv('design_GA_quadruples_1_results_blue.csv')
#d3 <- read.csv('design_GA_quadruples_2_results_blue.csv')



# example data with triads
# condition Factor 
# d1 <- read.csv('results/landscape/Samy/0_Factor.csv')
# d2 <- read.csv('results/landscape/Samy/1_Factor.csv')
# d3 <- read.csv('results/landscape/Samy/2_Factor.csv')
# d4 <- read.csv('results/landscape/Samy/3_Factor.csv')
# d5 <- read.csv('results/landscape/Samy/4_Factor.csv')
# d6 <- read.csv('results/landscape/Gabriel/0_Factor.csv')
# d7 <- read.csv('results/landscape/Gabriel/1_Factor.csv')
# d8 <- read.csv('results/landscape/Gabriel/2_Factor.csv')
# d9 <- read.csv('results/landscape/Gabriel/3_Factor.csv')
# d10 <- read.csv('results/landscape/Gabriel/4_Factor.csv')
# d11 <- read.csv('results/landscape/Muhammed/0_Factor.csv')
# d12 <- read.csv('results/landscape/Muhammed/1_Factor.csv')
# d13 <- read.csv('results/landscape/Muhammed/2_Factor.csv')
# d14 <- read.csv('results/landscape/Muhammed/3_Factor.csv')
# d15 <- read.csv('results/landscape/Muhammed/4_Factor.csv')
# d16 <- read.csv('results/landscape/Aleks/0_Factor.csv')
# d17 <- read.csv('results/landscape/Aleks/1_Factor.csv')
# d18 <- read.csv('results/landscape/Aleks/2_Factor.csv')
# d19 <- read.csv('results/landscape/Aleks/3_Factor.csv')
# d20 <- read.csv('results/landscape/Aleks/4_Factor.csv')

#  condition Compression
d1 <- read.csv('results/landscape/Samy/0_Compression.csv')
d2 <- read.csv('results/landscape/Samy/1_Compression.csv')
d3 <- read.csv('results/landscape/Samy/2_Compression.csv')
d4 <- read.csv('results/landscape/Samy/3_Compression.csv')
d5 <- read.csv('results/landscape/Samy/4_Compression.csv')
d6 <- read.csv('results/landscape/Gabriel/0_Compression.csv')
d7 <- read.csv('results/landscape/Gabriel/1_Compression.csv')
d8 <- read.csv('results/landscape/Gabriel/2_Compression.csv')
d9 <- read.csv('results/landscape/Gabriel/3_Compression.csv')
d10 <- read.csv('results/landscape/Gabriel/4_Compression.csv')
d11 <- read.csv('results/landscape/Muhammed/0_Compression.csv')
d12 <- read.csv('results/landscape/Muhammed/1_Compression.csv')
d13 <- read.csv('results/landscape/Muhammed/2_Compression.csv')
d14 <- read.csv('results/landscape/Muhammed/3_Compression.csv')
d15 <- read.csv('results/landscape/Muhammed/4_Compression.csv')
d16 <- read.csv('results/landscape/Aleks/0_Compression.csv')
d17 <- read.csv('results/landscape/Aleks/1_Compression.csv')
d18 <- read.csv('results/landscape/Aleks/2_Compression.csv')
d19 <- read.csv('results/landscape/Aleks/3_Compression.csv')
d20 <- read.csv('results/landscape/Aleks/4_Compression.csv')


# puts together. If you have only one file,
# then don't do this step, 
d <- rbind(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15,
           d16, d17, d18, d19, d20) #

# instead you just call  #d <- read.csv('myfile.csv')

# for quadruples, the data should have 5 columns
#    resp (observer response, 0 for S1-S2, 1 for S3-S4)
#    S1, S2, S3, S4: index of the stimulus values in each quadruple (integers).

# for triads, the data should have the same columns except S4.
#    the observer response resp is 0 for S1-S2, or 1 for S2-S3.

# we also need to know the actual stimulus values (not the indices). 
# In the case of the example of correlation in scatterplots,
# the stimulus vector is..

# I have just copied it from the generate_stim.py  script
# stim <- c(0, 2, 4, 8, 16)
stim <- c(0, 30, 60, 75, 85, 95)


# make a properly formatted dataframe. It adds an 'attribute' 
# called 'stimulus' to the DataFrame, so later MLDS keeps
# track of it and it can plot the x-axis properly.
if(ncol(d)==4){
  # for triads, use:
  df <- as.mlbs.df(d, stim)
}else{
  
  # bug correction / 14.12.2020
  # MLDS for quadruples requiere that always column S1 < column S2
  # and column S3 < column S4, even if order was reversed during the experiment.
  # my python script saves the order wrong, so the following code swaps
  # the indices when necessary
  temp_min <- pmin(d$S3, d$S4)
  d$S4 <- pmax(d$S3, d$S4)
  d$S3 <- temp_min
  
  temp_min <- pmin(d$S1, d$S2)
  d$S2 <- pmax(d$S1, d$S2)
  d$S1 <- temp_min
  ## end bug correction
  
  # for quadruples, use:
  df <- as.mlds.df(d, stim)
}



########################################################
## Analysis with MLDS 
# calls MLDS routine to estimate scale
scale <- mlds(df)


# just prints the scale values in the console
print(scale) 


## we can plot the perceptual scale and save the figure as PDF
# change the filename to your own name and include which method you used
pdf('scale_one_observer_one_condition.pdf')
plot(scale, xlab="Correlation coefficient - r", ylab="Perceptual scale")
dev.off()

# END

