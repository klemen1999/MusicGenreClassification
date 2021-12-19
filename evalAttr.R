########################################################################
#
# Ocenjevanje atributov v klasifikaciji in regresiji
#
########################################################################

evalAttr <- function(formula, data, measure, ...)
{
	df <- model.frame(formula, data)

	# mere za klasifikacijo zahtevajo disketen razred
	if (measure %in% c("InfGain", "GainRatio", "1-D", "Gini", "WE", "MDL", "RelifF", "J"))
	{ 
		if (!is.factor(df[,1]))
		{
			warning(paste("Izbrana ocena je definirana za klasifikacijo. Razred (", names(df)[1], ") ni kategoricna spremenljivka!", sep=""))
			
			df[,1] <- as.factor(df[,1])
		}
	}

	# mere za regresijo zahtevajo zvezen razred
	if (measure %in% c("ChangeOfVar", "RReliefF"))
	{ 
		if (is.factor(df[,1]))
		{
			warning(paste("Izbrana ocena je definirana za regresijo. Razred (", names(df)[1], ") ni zvezna spremenljivka!", sep=""))
			
			df[,1] <- as.numeric(df[,1])
		}
	}

	# nekatere mere ocenjujejo samo diskretne atribute
	if (measure %in% c("InfGain", "GainRatio", "1-D", "Gini", "WE", "MDL", "J", "ChangeOfVar"))
	{
		contatts <- vector()

		for (i in 2 : ncol(df))
		{
			if (!is.factor(df[,i]))
			{
				contatts <- c(contatts, i)
				df[,i] <- as.factor(df[,i])
			}
		}

		if (length(contatts) > 0)
		{
			problematic <- paste(names(df)[contatts], collapse=", ")
			warning(paste("Izbrana ocena je definirana samo za diskretne atribute. Nekateri atributi (", problematic, ") niso diskretni!", sep=""))
		}
	}

	if (measure == "InfGain")
		res <- calc.InfGain(df)
	else if (measure == "GainRatio")
		res <- calc.GainRatio(df)
	else if (measure == "1-D")
		res <- calc.OneMinusDist(df)
	else if (measure == "Gini")
		res <- calc.Gini(df)
	else if (measure == "WE")
		res <- calc.WE(df)
	else if (measure == "MDL")
		res <- calc.MDL(df)
	else if (measure == "ReliefF")
		res <- calc.ReliefF(df,...)
	else if (measure == "J")
		res <- calc.J.measure(df)
	else if (measure == "ChangeOfVar")
		res <- calc.Change.Of.Var(df)
	else if (measure == "RReliefF")
		res <- calc.RReliefF(df,...)
	else
		stop("Neznana ocena!")

	names(res) <- names(df)[-1]
	res
}


########################################################################
#
# Verjetnosti
#
########################################################################

#
# p(x)
#

prob <- function(x)
{
	t <- table(x)
	t/sum(t)
}


#
# p(xy)
#

prob.joint <- function(x, y)
{
	t <- table(x, y)
	t/sum(t)
}


#
# p(x|y)
#

prob.cond <- function(x, y)
{	
	t <- table(x, y)
	
	ty <- table(y)
	
	for (i in 1 : length(ty))
	{
		t[,i] <- t[,i] / ty[i]
	}

	t
}

#
# Laplaceov zakon zaporednosti
#

prob.laplace <- function(x, force.k = 0)
{
	t <- table(x)

	k <- force.k

	if (k == 0)
		k <- length(t)

	(t + 1) / (sum(t) + k)
}

#
# m-ocena verjetnosti p(x)
#

prob.m.estim <- function(x, m = 2, p0 = NULL)
{
	t <- table(x)

	if (is.null(p0)) p0 <- 1/length(t)

	(t + m * p0) / (sum(t) + m)
}

#
# m-ocena pogojne verjetnosti p(x|y)
#

prob.cond.m.estim <- function(x, y, p0 = NULL, m = 2)
{
	t <- table(x, y)
	t1 <- table(y)

	if (is.null(p0)) p0 <- 1/length(table(y))

	for (j in 1 : length(t1))
	{
		t[,j] <- (t[,j] + m * p0) / (t1[j] + m)
	}

	t
}

########################################################################
#
# Entropija
#
########################################################################

entropy <- function(x, y=NULL)
{
	if (is.null(y))
	{
		p <- prob(x)	
	}
	else
	{
		p <- prob.joint(x, y)	
	}

	-sum(p * log2(p), na.rm = TRUE)
}


########################################################################
#
# Ocene za klasifikacijo
#
########################################################################

#
# Info gain
#

calc.InfGain <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	res <- vector()

	Hr <- entropy(data[,1])

	for (i in 2 : ncol(data))
	{
		Ha <- entropy(data[,i])
		Hra <- entropy(data[,1], data[,i])
		res <- c(res, Hr + Ha - Hra)	
	}

	res
}


#
# Gain ratio
#

calc.GainRatio <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	res <- vector()

	Hr <- entropy(data[,1])

	for (i in 2 : ncol(data))
	{			
		Ha <- entropy(data[,i])
		Hra <- entropy(data[,1], data[,i])
			
		res <- c(res, (Hr + Ha - Hra) / Ha)	
	}

	res	
}


#
# 1-D
#

calc.OneMinusDist <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	res <- vector()

	Hr <- entropy(data[,1])

	for (i in 2 : ncol(data))
	{
		Ha <- entropy(data[,i])
		Hra <- entropy(data[,1], data[,i])
			
		res <- c(res, (Hr + Ha - Hra) / Hra)
	}

	res	
}


#
# Gini
#

calc.Gini <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	res <- vector()

	pk <- prob(data[,1])	
	sp2k <- sum(pk ^ 2)

	for (i in 2 : ncol(data))
	{
		pj <- prob(data[,i])
		pkj <- prob.cond(data[,1], data[,i])
		sp2kj <- apply(pkj ^ 2, 2, sum, na.rm=T)
	
		res <- c(res, sum(pj * sp2kj) - sp2k)
	}

	res		
}

#
# MDL ocena (diskretni atributi)
#

logbinom <- function(n, r)
{
	lfactorial(n)-lfactorial(r)-lfactorial(n-r)
}

logmultinom <- function(groups)
{
	lfactorial(sum(groups)) - sum(lfactorial(groups))	
}

prior.mdl <- function(class)
{
	t <- as.vector(table(class))

	n <- length(class)
	n0 <- length(t)

	(logmultinom(t) + logbinom(n + n0 - 1, n0 - 1)) / log(2)
}

post.mdl <- function(class, att)
{
	t <- as.matrix(table(class, att))
	n0 <- nrow(t)
	s <- 0

	for (i in 1:ncol(t))
	{
		nj <- sum(t[,i])
		s <- s + logmultinom(t[,i]) + logbinom(nj + n0 - 1, n0 - 1)
	}

	s / log(2)
}

mdl.att <- function(class, att)
{
	(prior.mdl(class) - post.mdl(class, att)) / length(class)
}


calc.MDL <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni
	
	res <- vector()

	for (i in 2 : ncol(data))		
		res <- c(res, mdl.att(data[,1], data[,i]))

	res		
}




#
# ReliefF
#

calc.dist <- function(att, rows, inst)
{
	#pripravimo vektor razdalj za vrednosti atributa po primerih
	dist <- rep(Inf, times = length(att))
	
	if (is.factor(att))
	{
		#izracunamo dejanske razdalje za izbrane primere
		dist[rows] <- att[rows] != att[inst]
	}
	else
	{
		#izracunamo dejanske razdalje za izbrane primere
		dist[rows] <- abs(att[rows] - att[inst]) / (max(att) - min(att))	
	}

	#vrnemo vektor razdalj po primerih
	dist
}

calc.ReliefF <- function(data, reliefIters = 100, reliefK = 5)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# ostali atributi so lahko diskretni ali zvezni

	n <- nrow(data)
	atts <- ncol(data)-1
	
	if (reliefIters == 0)
		reliefIters = n

	# verjetnosta distribucija razredov
	prob.classes <- prob(data[,1])
	
	# kvaliteta atributov
	w <- rep(0, times=atts)

	m <- min(reliefIters, n)

	# izberemo nakljucne ucne primere	
	inst <- sample(1:n, m, F)

	# za vse izbrane primere
	for (j in inst)
	{
		# razred opazovanega primera
		r.inst <- data[j, 1]

		# za vse vrednosti razreda
		for (razr in names(prob.classes))
		{
			# izberi vrstice, ki pripadajo trenutnem razredu
			sel <- (data[,1] == razr)
			
			# ne zelimo izbrati primera, ki ga obravnavamo
			sel[j] <- FALSE
			
			# pripravimo matriko razdalj
			dist.col = matrix(nrow = n, ncol = atts)
			
			# izracunamo razdalje za vse atribute
			for (a in 2:(atts+1))
			{
				dist.col[,a-1] <- calc.dist(data[,a], sel, j)
			}

			# sestejemo razdalje po atributih
			dist <- apply(dist.col, 1, sum)

			# k ne more biti vecji od stevila izbranih primerov
			k <- min(reliefK, sum(sel))
			
			# izberemo k najblizjih
			nearest.rows <- order(dist)[1:k]
	
			norm.factor <- m * k
			
			for (h in nearest.rows)
			{
				# normalizacija
				dist.col[h,] <- dist.col[h,] / norm.factor

				# ali imamo zadetek ali pogresek
				if (r.inst == razr)
				{
					w <- w - dist.col[h,]
				}
				else
				{
					prob.factor <- prob.classes[razr] / (1 - prob.classes[r.inst])
					w <- w + prob.factor * dist.col[h,]
				}
			}
		}	
	}

	w	
}


#
# Povprecna absolutna teza evidence
#

calc.WE <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	res <- vector()

	pk <- prob.laplace(data[,1])
	odds.k <- pk / (1-pk)

	for (i in 2 : ncol(data))
	{
		pj <- prob(data[,i])
		pkj <- prob.cond.m.estim(data[,1], data[,i])
		odds.kj <- pkj / (1-pkj)
	
		abs.log.odds <- abs(log(odds.kj / as.vector(odds.k)))

		we.k <- t(abs.log.odds)*as.vector(pj) 
		we.k <- apply(we.k, 2, sum)
		res <- c(res, sum(we.k * pk))
	}

	res		
}

#
# J-ocena
#

calc.J.measure <- function(data)
{
	# privzamemo, da je prvi stolpec diskreten razred
	# privzamemo, da so ostali atributi diskretni

	pk <- prob(data[,1])
	pk <- as.vector(pk)

	res <- list()

	for (i in 2 : ncol(data))
	{
		pj <- prob(data[,i])
		pkj <- prob.cond(data[,1], data[,i])
		tmp <- pkj * log2(pkj / pk)
		tmp <- apply(tmp, 2, sum, na.rm=T)
		jj <- pj * tmp
		names(jj) <- names(pj)
		
		res[[i-1]] <- jj
	}

	res
}

########################################################################
#
# Ocene za regresijske probleme
#
########################################################################

#
# Razlika variance
#

biasedVar <- function(x) 
{
	1/length(x) * sum((x-mean(x))^2)
}

calc.Change.Of.Var <- function(data)
{
	# privzamemo, da je prvi stolpec zvezen razred
	# privzamemo, da so ostali atributi diskretni

	
	res <- vector()

	# varianca zveznega razreda
	r.var <- biasedVar(data[,1])

	for (i in 2 : ncol(data))
	{
		pj <- prob(data[,i])

		j.var <- vector()
		for (j in names(pj))
		{
			sel <- data[,i] == j
			j.var <- c(j.var, biasedVar(data[sel,1]))
		}

		res <- c(res, r.var - sum(pj * j.var))
	}

	res
}


#
# RReliefF
#


calc.RReliefF <- function(data, reliefIters = 100, reliefK = 5)
{
	# privzamemo, da je prvi stolpec zvezen razred
	# ostali atributi so lahko diskretni ali zvezni

	n <- nrow(data)
	atts <- ncol(data)-1
	
	if (reliefIters == 0)
		reliefIters = n

	NdC <- 0
	NdA <- rep(0, times=atts)
	NdCdA <- rep(0, times=atts)

	m <- min(reliefIters, n)

	# izberemo nakljucne ucne primere	
	inst <- sample(1:n, m, F)

	# za vse izbrane primere
	for (j in inst)
	{
		sel <- rep(TRUE,times=n)
			
		# ne zelimo izbrati primera, ki ga obravnavamo
		sel[j] <- FALSE
			
		# pripravimo matriko razdalj
		dist.col = matrix(nrow = n, ncol = atts)
			
		# izracunamo razdalje za vse atribute
		for (a in 2:(atts+1))
		{
			dist.col[,a-1] <- calc.dist(data[,a], sel, j)
		}

		# sestejemo razdalje po atributih
		dist <- apply(dist.col, 1, sum)

		# k ne more biti vecji od stevila izbranih primerov
		k <- min(reliefK, sum(sel))
			
		# izberemo k najblizjih
		nearest.rows <- order(dist)[1:k]

		# razdalja med vrednostmi zveznega razreda
		dist.r <- calc.dist(data[,1], sel, j)
			
		for (h in nearest.rows)
		{
			NdC <- NdC + dist.r[h]

			NdA <- NdA + dist.col[h,]

			NdCdA <- NdCdA + dist.r[h] * dist.col[h,]
		}
	}
		
	NdC <- NdC / k
	NdA <- NdA / k
	NdCdA <- NdCdA / k

	NdCdA / NdC - (NdA - NdCdA) / (m - NdC)	
}



########################################################################
#
# Diskretizacija zveznih atributov
#
########################################################################

#
# Diskretizacija na podlagi enake sirine intervalov
#

disc.eq.width <- function(att, k=0)
{
	if (k == 0)
	{
		# hevristika za dolocanje stevila diskretnih vrednosti
		k <- nclass.scott(att)
	}

	# diskretiziramo vektor att na k diskretnih vrednosti
	cut(att, k)
}

#
# Diskretizacija na podlagi enake zastopanosti intervalov
#

disc.eq.freq <- function(att, k) 
{
	breaks <- quantile(att, probs = seq(0, 1, length.out = k+1), na.rm = TRUE)

	if(any(duplicated(breaks)))
		warning(paste("Vrednosti atributa so diskretizirane na manj kot",k,"intervalov!"))
 
	breaks <- unique(breaks)
	breaks[1] <- -Inf
	breaks[length(breaks)] <- Inf

	cut(att, breaks)
}

#
# Diskretizacija na podlagi principa MDL
#

disc.mdlp <- function(att, class)
{
	indices <- order(att)

	b <- cbind(
		min(att, na.rm=T) - 1,
		disc.mdlp.rec(att[indices], class[indices]),
		max(att, na.rm=T))
	
	cut(att, breaks = b)	
}

disc.mdlp.rec <- function(att, class)
{
	#
	# privzamemo, da je vektor att sortiran
	#
	
	b <- vector()

	s <- length(att)

	if (s > 1)
	{
		Hs <- entropy(class)
		k <- length(unique(class))
		max.gain <- 0
		i <- 1

		repeat
		{
			i <- which(att > att[i])[1]
			if (is.na(i))
				break

			H1 <- entropy(class[1:(i-1)])
			H2 <- entropy(class[i : s])
			gain <- Hs - (i-1) / s * H1 - (s - i + 1)/s * H2

			if (gain >= max.gain)
			{
				max.gain <- gain
				s1 <- i - 1
				Hs1 <- H1
				Hs2 <- H2
			}
		}

		if (max.gain > 0)
		{
			#MDL
			k1 <- length(unique(class[1:s1]))
			k2 <- length(unique(class[(s1 + 1) : s]))

			th <- (log2(s - 1) + log2(3 ^ k - 2) - k * Hs + k1 * Hs1 + k2 * Hs2) / s
	
			if (max.gain > th)
			{
				b <- cbind(
					disc.mdlp.rec(att[1:s1], class[1:s1]),
					att[s1],
					disc.mdlp.rec(att[(s1 + 1) : s], class[(s1 + 1) : s]))
			}
		}
	}

	b
}

########################################################################
#
# Binarizacija zveznih in nominalnih atributov glede na razred
#
########################################################################

binarize <- function(att, class)
{
	if (!is.factor(class))
	{
		warning("Razred ni faktor!")
		return (att)
	}

	if (nlevels(class) < 2)
	{
		warning("Razred ima premalo razlicnih vrednosti!")
		return (att)
	}

	if (is.numeric(att))
	{
		do.binarize.numeric(att, class)
	}
	else if (is.factor(att))
	{
		if (nlevels(att) < 3)
		{
			warning("Atribut ima premalo razlicnih vrednosti!")
			return (att)
		}
		else if (nlevels(att) > 15)
		{
			warning("Atribut ima prevec razlicnih vrednosti!")
			return (att)
		}

		do.binarize.nominal(att, class)
	}
	else
	{
		warning("Neustrezna oblika atributa!")
		return (att)
	}
}

do.binarize.numeric <- function(att, class)
{
	#
	# privzamemo, da ima class vsaj dve razlicni vrednosti
	#
 
	origatt <- att

	indices <- order(att)
	att <- att[indices]
	class <- class[indices]

	attvals <- sort(unique(att))

	if (length(attvals) < 2)
	{
		warning("Attribut ima eno samo vrednost!")
		return (origatt)
	}

	bestres <- -Inf
	bestCutVal <- Inf
		
	Hr <- entropy(class)

	splits <- which(diff(as.integer(class)) != 0)
	for (i in splits)
	{
		v1 <- att[i]
		j <- which(attvals == v1)
		if (j < length(attvals))
			v2 <- attvals[j+1]
		else
			v2 <- attvals[j-1]

		cutVal <- (v1+v2)/2

		binatt <- cut(att, breaks=c(-Inf, cutVal, Inf))

		Ha <- entropy(binatt)
		Hra <- entropy(class, binatt)
		res <- Hr + Ha - Hra

		if (res > bestres)
		{
			bestres <- res
			bestCutVal <- cutVal
		}
	}

	cut(origatt, breaks=c(-Inf, bestCutVal, Inf))		
}

do.binarize.nominal <- function(att, class)
{
	#
	# privzamemo, da ima class vsaj dve razlicni vrednosti, 
	# ter da ima att vsaj 3 in ne vec kot 15 razlicnih vrednosti
	#

	Hr <- entropy(class)

	attVals <- levels(att)
	n <- nlevels(att)

	bestres <- -Inf
	bestSubset <- NULL

	subset <- rep(F, n)
	m <- 2^(n-1)-1

	for (j in 1:m)
	{
		for (i in 1:length(subset))	
		{
			if (!subset[i])
			{
				subset[i] <- T
				break
			}
			else
				subset[i] <- F
		}

		sel <- attVals[subset]

		binatt <- ifelse(att %in% sel, "0", "1")

		Ha <- entropy(binatt)
		Hra <- entropy(class, binatt)
		res <- Hr + Ha - Hra

		if (res > bestres)
		{
			bestres <- res
			bestSubset <- subset
		}
	}

	sel <- attVals[bestSubset]
	val1 <- paste(sel, collapse=",")
	val2 <- paste(attVals[!bestSubset], collapse=",")
	factor(ifelse(att %in% sel, val1, val2))
}

