## script to import csv data, calculate, & plot Gini indices through time
## for report submitted to Science 8-'17 (make that Nature)

# setwd("/Volumes/G-RaidStudio/Gini-Amerind/Gini_Vol_Chapt_11/Gini_Data_R_Scripts_Chapt_11")

devtools::install_github("bocinsky/FedData");library(FedData,quietly=T)

pkg_test("DescTools")
pkg_test("ggplot2")
pkg_test("gridExtra")
pkg_test("grid")
pkg_test("data.table")
pkg_test("dplyr")
pkg_test("magrittr")
pkg_test("lattice")
pkg_test("RColorBrewer")
pkg_test("ggthemes")
pkg_test("tidyr")
pkg_test("devtools")
pkg_test("mgcv")
pkg_test("EnvStats")
pkg_test("gmodels")
pkg_test("ggrepel")
pkg_test("ineq")

library(dplyr)
library(tidyverse)
library(data.table)
library(tidyr)
library(ineq)
# create various useful datasets
Gini_Tenoch <- read.csv(file="Tenochelites.csv",head=TRUE,sep=",")
mean(Gini_Tenoch[[2]])
sd(Gini_Tenoch[[2]])
# elites mean = 1276.667, sd = 167.455

Gini_Tenoch1 <- read.csv(file="Tenoch_WC.csv",head=TRUE,sep=",") # Wealthy Commoners
mean(Gini_Tenoch1[[2]])
sd(Gini_Tenoch1[[2]])
# Wealthy Commoners mean = 811.333, sd = 9.815

Gini_Tenoch2 <- read.csv(file="Tenoch_commoners.csv",head=TRUE,sep=",") # Commoners
mean(Gini_Tenoch2[[2]])
sd(Gini_Tenoch2[[2]])
# Commoners mean = 251.793, sd = 143.399

# CHANGED TO MATCH ANGELA 2020.7 DATASET 
l <- c()
for (i in c(1:1000)) {
  
  Emp <- rnorm(1, mean = 25425, sd = 0) # Xalla
  Elites <- rnorm(373, mean = 1645.6, sd = 485.7) # high
  WC <- rnorm(13430, mean = 348.5, sd = 217.5) # Intermediate
  C <- rnorm(545, mean = 25, sd = 5) # low status
  Uncertain <- rnorm(1146, mean = 348.5, sd = 217.5)
  # C <- ifelse (C < 52, 52, C) # put as lower bound the mean of 5 smallest commoner houses
  
  Emp <- data.frame(Emp)
  Elites <- data.frame(Elites)
  WC <- data.frame(WC)
  C <- data.frame(C)
  Uncertain <- data.frame(Uncertain)
  
  Emp <- rename(Emp, size = Emp)
  Elites <- rename(Elites, size = Elites)
  WC <- rename(WC, size = WC)
  C <- rename(C, size = C)
  Uncertain <- rename(Uncertain, size = Uncertain)
  
  Trial1 <- rbind(Emp,Elites,WC,C, Uncertain) 
  l <- c(l, ineq(Trial1$size,type="Gini")) 

}
mean(l)
sd(l)

# outputs 0.351, 0.348, 0.351, 0.350, 0.351, 0.350
Gini <- c(0.351, 0.348, 0.351, 0.350, 0.351, 0.350)

sd(Gini)
mean(Gini)
# outputs 0.350 and 0.001