function stochastic_optimizers()
close all
fsz = 20;
%% Pick the number of PCAs for the representation of images
nPCA = 20;
%%
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;    
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
%% use PCA to reduce dimensionality of the problem to 20
[d1,d2,~] = size(train1);
X1 = zeros(n1train,d1*d2);
X2 = zeros(n2train,d1*d2);
for j = 1 : n1train
    aux = train1(:,:,j);
    X1(j,:) = aux(:)';
end
for j = 1 :n2train
    aux = train2(:,:,j);
    X2(j,:) = aux(:)';
end
X = [X1;X2];
D1 = 1:n1train;
D2 = n1train+1:n1train+n2train;
[U,Sigma,~] = svd(X','econ');
% esort = diag(Sigma);
% figure;
% plot(esort,'.','Markersize',20);
% grid;
Xpca = X*U(:,1:nPCA); % features
%% split the data to training set and test set
Xtrain = Xpca;
Ntrain = n1train + n2train;
Xtest1 = zeros(n1test,d1*d2);
for j = 1 : n1test
    aux = test1(:,:,j);
    Xtest1(j,:) = aux(:)';
end
for j = 1 :n2test
    aux = test2(:,:,j);
    Xtest2(j,:) = aux(:)';
end
Xtest = [Xtest1;Xtest2]*U(:,1:nPCA);
%% category 1 (1): label 1; category 2 (7): label -1
label = ones(Ntrain,1);
label(n1train+1:Ntrain) = -1;
%X is Ntrain x Ndim:
%row i of X:  [ ith training example ]
dim = nPCA;
%% (1) Log loss function

lam = 0.001; % Tikhonov regularization parameter

fun = @(I,w)qloss(I,Xtrain,label,w,lam);
gfun = @(I,w)qlossgrad(I,Xtrain,label,w,lam);
gfun2 = @(I,w)qlossgrad2(I,Xtrain,label,w,lam);
Hvec = @(I,w,v)Hvec0(I,Y,w,v,lam);
%%
bsz = 1000;
tol = 1e-4;
% kmax = 1e3*num_batches; % the max number of iterations
% kmax = 10000;
max_epochs = 100;
w = ones(dim^2 + dim + 1, 1);
%%
tic;
% [w,f,gnorm] = my_LBFGS(fun, gfun, Xtrain,label,w, bsz, kmax, tol);
% method = 'LBFGS';

% [w,f,gnorm] = ADAM(fun, gfun, gfun2, Hvec, Xtrain,label,w, bsz, kmax, tol);
% method = 'ADAM';

% [w,f,gnorm] = NGD(fun, gfun, Hvec, Xtrain,label,w, bsz, kmax, tol);
% method = 'NGD';

[w,f,gnorm, iter] = SGD(fun, gfun, Hvec, Xtrain,label,w, bsz, max_epochs, tol);
runtime = toc;
method = 'SGD';

% plot the objective function
fig = figure;
plot(f,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('f','fontsize',fsz);
xlim([0, size(f,2)]);
set(gca,'fontsize',fsz,'Yscale','log');
fname = sprintf('fval_%s_bsz%d.png', method, bsz);
saveas(fig, fname);

% plot the norm of the gradient
fig = figure;
plot(gnorm,'Linewidth',2);
xlabel('iter','fontsize',fsz);
ylabel('||g||','fontsize',fsz);
xlim([0, size(gnorm,2)]);
set(gca,'fontsize',fsz,'Yscale','log');
fname = sprintf('gnorm_%s_bsz%d.png', method, bsz);
saveas(fig, fname);
%% apply the results to the test set
Ntest = n1test+n2test;
testlabel = ones(Ntest,1);
testlabel(n1test+1:Ntest) = -1;
I = 1:Ntest;
test = myquadratic(Xtest,testlabel,I,w);

hits = find(test > 0);
misses = find(test < 0);
nhits = length(hits);
nmisses = length(misses);
fprintf('n_correct = %d, n_wrong = %d, accuracy %d percent\n',nhits,nmisses,nhits/Ntest);
fprintf('final_loss = %d, gnorm = %d, iterations %d, time %d, accruacy %d\n', f(end), gnorm(end), iter, runtime, nhits/Ntest);

end
%%
%%
%%
function f = qloss(I,Xtrain,label,w,lam)
f = sum(log(1 + exp(-myquadratic(Xtrain,label,I,w))))/length(I) + 0.5*lam*w'*w;
end

function f = lsq_loss(I,Xtrain,label,w)
R = log(1 + exp(-myquadratic(Xtrain,label,I,w)));
f = 0.5*sum(R.^2);
end

%%
function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end

function g = qlossgrad2(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
ya2 = (ya).^2;
X2 = X.^2;
qterm = X2'*((ya2*ones(1,d)).*X2);
lterm = X2'*ya2;
sterm = sum(ya2);
g = [qterm(:);lterm;sterm]/length(I) + (lam*w).^2;
end

%%
function q = myquadratic(Xtrain,label,I,w)
X = Xtrain(I,:);
[N, d] = size(X);
d2 = d^2;
y = label(I);
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);

qterm = zeros(N,1);
for i=1:N
    qterm(i) = y(i)*X(i,:)*W*X(i,:)';
end
q = qterm + ((y*ones(1,d)).*X)*v + y*b;
end


