function XY = D(X,Y)

XY = X.*log(X./(Y+eps))-X+Y;