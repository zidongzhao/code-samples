require(ppcor)
require(psych)
require(ggplot2)
require(rlm)
require(moments)
require(boot)
require(effsize)
require(reshape2)
require(ggplot2)
require(gridExtra)
require(dplyr)
require(stringr)
require(tidyr)
require(text2vec)
require(heatmaply)
require(lmerTest)
require(lavaan)
require(rbin)
require(Rmisc)
require(mixtools)
require(MASS)
require(faux)
# load('wave1_out')

# rewriting the original simulated correlation so nothing is anchored on empirical data ------
set.seed(20200604)
steps <- seq(0,1,.005)
iter.per.step <- nrow(df.indiff)
niter <- 10
r.super <- rep(NA,length(steps)*niter) # correlation between two correlations
r.super.split <- rep(NA,length(steps)*niter)
for(i in 1:length(steps)){
  r.base.1 <- rep(NA, iter.per.step)
  r.base.2 <- rep(NA, iter.per.step)
  r.base.3 <- rep(NA, iter.per.step)
  r.base.4 <- rep(NA, iter.per.step)
  for(k in 1:niter){
    for(j in 1:iter.per.step){
      anchor <- as.vector(scale(rnorm(75)))
      correlated.vec <- rnorm_pre(anchor, r = steps[i],empirical = T) # vary egocentrism
      third.vec <- rnorm(75) # vector to correlate with both the anchor and the correlated vec
      r.base.1[j] <- cor(anchor,third.vec)
      r.base.2[j] <- cor(correlated.vec,third.vec)
      # step 1. split half
      ind.all <- seq(1,75)
      half.ab <- sample(ind.all,sample(c(37,38),1),replace = F)
      half.bc <- ind.all[!(ind.all %in% half.ab)]
      r.base.3[j] <- cor(anchor[half.ab],third.vec[half.ab])
      r.base.4[j] <- cor(anchor[half.bc],third.vec[half.bc])
    }
    cormat <- cor(apply(cbind(r.base.1,r.base.2),2,fisherz))
    r.super[(i-1)*niter+k] <- cormat[2]
    cormat2 <- cor(apply(cbind(r.base.3,r.base.4),2,fisherz))
    r.super.split[(i-1)*niter+k] <- cormat2[2]
  }
  print(i)
}
test_ind <- rep(steps,each = niter)
test_dat <- as.data.frame(cbind(test_ind,r.super, r.super.split))
# graph without splitting procedure
g.rrabrac.varyac <- ggplot(data = test_dat, aes(x = test_ind, y = r.super)) +
  geom_point(alpha = .3)+
  xlab('Correlation between A and C')+
  ylab('Secondary correlation between rAB and rBC') +
  ggtitle('Secondary correlation as a function of the correlation between unshared base variable')
# graph when the the splitting procudure is employed
g.rrabrac.varyac.split <- ggplot(data = test_dat, aes(x = test_ind, y = r.super.split)) +
  geom_point(alpha = .3)+
  xlab('Correlation between A and C')+
  ylab('Secondary correlation between rAB and rBC') +
  ggtitle('Secondary correlation as a function of the correlation between unshared base variable')

# full generative----
# first generate top level correlation, given the correlation of correlations
# then for each person generate correlated base variables
# look at top level param recovery for split vs non-split
# specify params
cor_cor <- .8
split_recover <- function(cor_cor,ac){
  # generate top level correlation - draw from mvnorm then normalize and shift to between 0 and 1
  top_cor <- mvrnorm(n = 102,mu = c(0,0), Sigma = matrix(c(1,cor_cor,cor_cor,1),2,2))
  top_cor <- (top_cor-min(top_cor))/(max(top_cor)-min(top_cor))
  # remove 0s and 1s
  top_cor <- top_cor[!(top_cor[,1] == 0 | top_cor[,2] == 0 | top_cor[,1] == 1 | top_cor[,2] == 1),]
  # given these top level correlation, generate 3 base variables
  # 2 (ab,bc) of the base correlations are already determined, third (ac) can be fixed or varied systematically
  # here we first try fixing it at .5
  top_recover <- matrix(0,nrow = dim(top_cor)[1],ncol = dim(top_cor)[2])
  for(i in 1:dim(top_cor)[1]){
    ab <- top_cor[i,1]
    bc <- top_cor[i,2]
    ac <- .5
    sig <- matrix(c(1,ab,ac,
                    ab,1,bc,
                    ac,bc,1),3,3)
    # generate and scale to 0-100
    base_vars <- mvrnorm(n = 75,mu = c(0,0,0), Sigma = sig,2,2)
    base_vars <- (base_vars-min(base_vars))/(max(base_vars)-min(base_vars)) * 100
    # split and attempt recovery
    ind.all <- seq(1,75)
    half.ab <- sample(ind.all,sample(c(37,38),1),replace = F)
    half.bc <- ind.all[!(ind.all %in% half.ab)]
    top_recover[i,1] <- cor(base_vars[half.ab,1],base_vars[half.ab,2])
    top_recover[i,2] <- cor(base_vars[half.bc,2],base_vars[half.bc,3])
  }
  return(cor(top_recover[,1],top_recover[,2]))
}
# grid + simulate
set.seed(65284739)
niter <- 250
corcor_steps <- seq(.1,.9,.2)
ac_steps <- seq(.2,.8,.2)
df.split.recover <- as.data.frame(matrix(NA,nrow = niter,
                                         ncol = length(corcor_steps)*length(ac_steps)))
col.counter <- 1
for(corcor in corcor_steps){
  for(ac in ac_steps){
    recovered_corcor <- rep(0,niter)
    for(i in 1:niter){
      recovered_corcor[i] <- split_recover(corcor,ac)
    }
    df.split.recover[,col.counter] <- recovered_corcor
    col.lab <- paste('corcor_',corcor,'_ac_',ac,sep='')
    colnames(df.split.recover)[col.counter] <- col.lab
    col.counter <- col.counter + 1
  }
}
df.split.recover <- melt(df.split.recover)
newcols <- str_split_fixed(df.split.recover$variable,'_',n=4)
df.split.recover$corcor <- newcols[,2]
df.split.recover$ac <- newcols[,4]
g.split.recover <- ggplot(data = df.split.recover) + geom_histogram(aes(x = value)) + facet_grid(rows = vars(corcor), cols = vars(ac))


# two layer bootstrapping using real data -------
  # layer one: random split of states into similarity and accuracy measures
  # layer two: within the random split, bootstrap n times over participants

# write this as a function so it can be used on more than similarity/accuracy
# function takes in raw long dataframe (dfuse/dfuuu), strings indicating variables, and niters at the item and the subject levels
random.split.bootstrap <- function(rawdata,shared,vec1,vec2,steps.l1,steps.l2){
  r.super <- rep(NA, steps.l1 * steps.l2)
  for(i in 1:steps.l1){
    #generate a random assignment of the items into the two constructs: e.g. similarity and accuracy
    nitem <- length(unique(rawdata$prompt_num)) #column name hard coded
    ind.all <- seq(1,nitem)
    small.half <- floor(nitem/2) # odd number of items
    big.half <- nitem - small.half
    half.ab <- sample(ind.all,sample(c(big.half,small.half),1),replace = F)
    half.bc <- ind.all[!(ind.all %in% half.ab)]
    # calculate the two base ccorrelations for all subjects given the random split
    r.base.1 <- rep(NA, length(unique(rawdata$sub_num)))
    r.base.2 <- rep(NA, length(unique(rawdata$sub_num)))
    counter <- 1
    for(q in unique(rawdata$dyad_num)){
      dfind1 <- rawdata[which(rawdata$dyad_num == q & rawdata$ind_num == 1),]
      dfind2 <- rawdata[which(rawdata$dyad_num == q & rawdata$ind_num == 2),]
      #base correlation 1 for both sub in the dyad
      r.b1.1 <- cor(dfind1[half.ab,shared],dfind1[half.ab,vec1])
      r.b1.2 <- cor(dfind2[half.ab,shared],dfind2[half.ab,vec1])
      #base correlation 2 for both sub in the dyad
      r.b2.1 <- cor(dfind1[half.bc,shared],dfind1[half.bc,vec2])
      r.b2.2 <- cor(dfind2[half.bc,shared],dfind2[half.bc,vec2])
      r.base.1[c(counter, counter+1)] <- c(r.b1.1,r.b1.2)
      r.base.2[c(counter, counter+1)] <- c(r.b2.1,r.b2.2)
      counter <- counter + 2
    }
    zr.base.1 <- fisherz(r.base.1)
    zr.base.2 <- fisherz(r.base.2)
    # given the random split, do a small number of bootstrapping across subjects
    r.boot.sub <- rep(NA, steps.l2)
    for(j in 1:steps.l2){
      # for each iteration, randomly sample dyads with replacement
      temp.subs <- sample(1:length(zr.base.1),replace = T) #column name hard coded
      temp.base.1 <- zr.base.1[temp.subs]
      temp.base.2 <- zr.base.2[temp.subs]
      r.boot.sub[j] <- cor(temp.base.1,temp.base.2)
    }
    # print(r.boot.sub)
    r.super[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- r.boot.sub
  }
  return(r.super)
}

# bootstrapping relation between similarity and accuracy ----
steps.l1 <- 1000
steps.l2 <- 10
set.seed(29187562)
sim.acc.split.bootstrap <- random.split.bootstrap(dfuuu,
                                                  shared = '',
                                                  vec1 = '',
                                                  vec2 = '',
                                                  steps.l1 = steps.l1,
                                                  steps.l2 = steps.l2)
ci.sim.acc.split.bootstrap <- quantile(sim.acc.split.bootstrap,c(.025,.975))

##### next step: mediation models? higher sim2a -> higher egoc -> higher accuracy? -----
#function for calculating geometric mean
gm_mean = function(x, na.rm=TRUE, zero.propagate = FALSE){
  if(any(x < 0, na.rm = TRUE)){
    return(NaN)
  }
  if(zero.propagate){
    if(any(x == 0, na.rm = TRUE)){
      return(0)
    }
    exp(mean(log(x), na.rm = na.rm))
  } else {
    exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
  }
}
# mediation 1: sim2a -> egoc -> accuracy ----
steps.l1 <- 50
steps.l2 <- 50
# initializing storage
med.s2a.a <- rep(NA, steps.l1 * steps.l2)
med.s2a.b <- rep(NA, steps.l1 * steps.l2)
med.s2a.c <- rep(NA, steps.l1 * steps.l2)
med.s2a.ab <- rep(NA, steps.l1 * steps.l2)
med.s2a.total <- rep(NA, steps.l1 * steps.l2)
med.s2a.a.p <- rep(NA, steps.l1 * steps.l2)
med.s2a.b.p <- rep(NA, steps.l1 * steps.l2)
med.s2a.c.p <- rep(NA, steps.l1 * steps.l2)
med.s2a.ab.p <- rep(NA, steps.l1 * steps.l2)
med.s2a.total.p <- rep(NA, steps.l1 * steps.l2)

rawdata <- dfuuu
# specifying the mediation model in lavaan
med <- ' # direct effect
             acc ~ c*s2a
           # mediator
             egoc ~ a*s2a
             acc ~ b*egoc
           # indirect effect (a*b)
             ab := a*b
           # total effect
             total := c + (a*b)
         '
set.seed(57482932)
for(i in 1:steps.l1){
  #generate a random assignment of the items into the two constructs: e.g. similarity and accuracy
  nitem <- length(unique(rawdata$prompt_num)) #column name hard coded
  ind.all <- seq(1,nitem)
  small.half <- floor(nitem/2) # odd number of items
  big.half <- nitem - small.half
  half.ab <- sample(ind.all,sample(c(big.half,small.half),1),replace = F)
  half.bc <- ind.all[!(ind.all %in% half.ab)]
  # calculate the two base ccorrelations for all subjects given the random split
  r.base.ps2a <- rep(NA, length(unique(rawdata$sub_num))) #base
  r.base.egoc <- rep(NA, length(unique(rawdata$sub_num)))
  r.base.acc <- rep(NA, length(unique(rawdata$sub_num)))
  counter <- 1
  for(q in unique(rawdata$dyad_num)){
    dfind1 <- rawdata[which(rawdata$dyad_num == q & rawdata$ind_num == 1),]
    dfind2 <- rawdata[which(rawdata$dyad_num == q & rawdata$ind_num == 2),]
    dfrest <- rawdata[which(rawdata$dyad_num != q),]
    #base correlation accuracy for both sub in the dyad
    r.bacc.1 <- cor(dfind1[half.ab,'friend_self'],dfind1[half.ab,'self_target']) # var a and c in the mediation will share base variable indices
    r.bacc.2 <- cor(dfind2[half.ab,'friend_self'],dfind2[half.ab,'self_target'])
    # base correlation similarity to group
    r.bs2a.1 <- cor(dfind1[half.ab,'self_self'], dfind1[half.ab,'ptonavg.excludedyad'])
    r.bs2a.2 <- cor(dfind2[half.ab,'self_self'], dfind2[half.ab,'ptonavg.excludedyad'])
    # base correlation egocentrism
    r.begoc.1 <- cor(dfind1[half.bc,'self_self'], dfind1[half.bc,'self_target'])
    r.begoc.2 <- cor(dfind2[half.bc,'self_self'], dfind2[half.bc,'self_target'])
    # tally
    r.base.ps2a[c(counter, counter+1)] <- c(r.bs2a.1,r.bs2a.2)
    r.base.acc[c(counter, counter+1)] <- c(r.bacc.1,r.bacc.2)
    r.base.egoc[c(counter, counter+1)] <- c(r.begoc.1,r.begoc.2)
    counter <- counter + 2
  }
  zr.base.ps2a <- fisherz(r.base.ps2a)
  zr.base.acc <- fisherz(r.base.acc)
  zr.base.egoc <- fisherz(r.base.egoc)
  # given the random split, do a small number of bootstrapping across subjects
  t.a <- rep(NA, steps.l2)
  t.b <- rep(NA, steps.l2)
  t.c <- rep(NA, steps.l2)
  t.ab <- rep(NA, steps.l2)
  t.total <- rep(NA, steps.l2)
  t.a.p <- rep(NA, steps.l2)
  t.b.p <- rep(NA, steps.l2)
  t.c.p <- rep(NA, steps.l2)
  t.ab.p <- rep(NA, steps.l2)
  t.total.p <- rep(NA, steps.l2)
  for(j in 1:steps.l2){
    # for each iteration, randomly sample dyads with replacement
    temp.subs <- sample(1:length(zr.base.ps2a),replace = T) #column name hard coded
    temp.base.ps2a <- scale(zr.base.ps2a[temp.subs])
    temp.base.acc <- scale(zr.base.acc[temp.subs])
    temp.base.egoc <- scale(zr.base.egoc[temp.subs])
    dt.med <- data.frame(s2a = temp.base.ps2a, acc = temp.base.acc, egoc = temp.base.egoc)
    fit.med <- sem(med, data = dt.med)
    t.a[j] <- coef(fit.med,'user')['a']
    t.b[j] <- coef(fit.med,'user')['b']
    t.c[j] <- coef(fit.med,'user')['c']
    t.ab[j] <- coef(fit.med,'user')['ab']
    t.total[j]  <- coef(fit.med,'user')['total']
    pe <- lavaan::parameterestimates(fit.med)
    t.a.p[j] <- pe$pvalue[pe$label == 'a']
    t.b.p[j] <- pe$pvalue[pe$label == 'b']
    t.c.p[j] <- pe$pvalue[pe$label == 'c']
    t.ab.p[j] <- pe$pvalue[pe$label == 'ab']
    t.total.p[j] <- pe$pvalue[pe$label == 'total']
  }
  print(i)
  med.s2a.a[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.a
  med.s2a.b[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.b
  med.s2a.c[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.c
  med.s2a.ab[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.ab
  med.s2a.total[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.total
  med.s2a.a.p[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.a.p
  med.s2a.b.p[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.b.p
  med.s2a.c.p[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.c.p
  med.s2a.ab.p[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.ab.p
  med.s2a.total.p[((i-1) * steps.l2 + 1):((i-1) * steps.l2 + j)] <- t.total.p
}

#SAM goodness, estimates of interest are random effect correlations.
goodstuff <- lmer(data = dfuuu,
                  self_target ~ friend_self + ptonavg.excludedyad + self_self +
                    (1+friend_self + ptonavg.excludedyad + self_self|dyad_num/sub_num)+
                    (1|prompt_num))
