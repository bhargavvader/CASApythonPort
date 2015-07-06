function [k1,kn,err] = decompose_kernel(h_orig)
% This function does the decomposition of a separable nD kernel into
% its 1D components, such that a convolution with each of these
% components yields the same result as a convolution with the full nD
% kernel, at a drastic reduction in computational cost.
%
% SYNTAX:
% =======
%    [K1,KN,ERR] = DECOMPOSE_KERNEL(H)
% computes a set of 1D kernels K1{1}, K1{2}, ... K1{N} such that the
% convolution of an image with all of these in turn yields the same
% result as a convolution with the N-dimensional input kernel H:
%    RES1 = CONVN(IMG,H);
%    RES2 = IMG;
%    FOR II=1:LENGTH(K1)
%       RES2 = CONVN(RES2,K1{II});
%    END
%
% KN is the reconstruction of the original kernel H from the 1D
% kernels K1, and ERR is the sum of absolute differences between H
% and KN.
% The syntax mimics Dirk-Jan Kroon's submission to the FileExchange
% (see below).
%
% EXPLANATION:
% ============
%
% In general, for a 2D kernel H, the convolution with 2D image F:
%    G = F * H
% is identical to the convolution of the image with column vector H1
% and convolution of the result with row vector H2:
%    G = ( F * H1 ) * H2   .
% In MATLAB speak, this means that
% > CONV2(F,H) == CONV2(CONV2(F,H1),H2)
%
% Because of the properties of the convolution,
%    ( F * H1 ) * H2 = F * ( H1 * H2 )   ,
% meaning that the convolution of the two 1D filters with each other
% results in the original filter H. And because H1 is a column vector
% and H2 a row vector,
%    H = H1 * H2 = H1 H2   .
% Thus, we need to find two vectors whose product yields the matrix H.
% In MATLAB speak we need to solve the equation
% > H1*H2 == H
%
% The function in the standard MATLAB toolbox, FILTER2, does just
% this, and it does it using singular value decomposition:
%    U S V' = H   ,
%    H1(i) = U(i,1)  S(1,1)^0.5   ,
%    H2(i) = V(i,1)* S(1,1)^0.5   .  (the * here is the conjugate!)
%
% Note that, if the kernel H is separable, all values of S are zero
% except S(1,1). Also note that this is an under-determined problem,
% in the sense that
%    H = H1 H2 = ( a H1 ) ( 1/a H2 )   ;
% that is, it is possible to multiply one 1D kernel with any value
% and compensate by dividing the other kernel with the same value.
% Our solution will, in effect, just choose one of the infinite number
% of (equivalent) solutions.
%
% To extend this concept to nD, what we need to understand is that it
% is possible to collapse all dimensions except one, obtaining a 2D
% matrix, and solve the above equation. This results in a 1D kernel
% and an (n-1)D kernel. Repeat the process until all you have is a
% set of 1D kernels and you're done!
%
% This function is inspired by a solution to this problem that
% Dirk-Jan Kroon posted on the File Exchange recently:
% http://www.mathworks.com/matlabcentral/fileexchange/28218-separate-kernel-in-1d-kernels
% His solution does the whole decomposition in one go, by setting up
% one big set of equations. He noted a problem with negative values,
% which produce complex 1D kernels. The magnitude of the result is
% correct, but the sign is lost. He needs to resort to some heuristic
% to determine the sign of each element. What he didn't notice (or
% didn't mention) is the problem that his solution has with 0 values.
% The SVD solution doesn't have this problem, although it sometimes
% does produce a slightly worse solution. For example, in the first
% example below, Dirk-Jan Kroon's solution is exact, whereas this one
% produces a very small error. Where Dirk-Jan Kroon's solution cannot
% find the exact solution, this algorithm generally does better.
% 
% EXAMPLES:
% =========
%
% Simplest 5D example:
%
%    H = ones(5,7,4,1,5);
%
%    [K1,~,err] = SeparateKernel(H); % D.Kroon's submission to FileEx.
%    err
%
%    [k1,~,err] = decompose_kernel(H);
%    err
%
% 2D example taken from Dirk-Jan Kroon's submission:
%
%    a = permute(rand(4,1),[1 2 3])-0.5;
%    b = permute(rand(4,1),[2 1 3])-0.5;
%    H = repmat(a,[1 4]).*repmat(b,[4 1]);
%
%    [K1,~,err] = SeparateKernel(H);
%    err
%
%    [k1,~,err] = decompose_kernel(H);
%    err
%
% 2D example for which Dirk-Jan Kroon's solution has problems:
%
%    H = [1,2,3,2,1]'*[1,1,3,0,3,1,1];
%   
%    [K1,~,err] = SeparateKernel(H);
%    err
%   
%    [k1,~,err] = decompose_kernel(H);
%    err
%
% 3D example that's not separable:
%
%    H = rand(5,5,3);
%
%    [K1,~,err] = SeparateKernel(H);
%    err
%
%    [k1,~,err] = decompose_kernel(H);
%
% Example to apply a convolution using the decomposed kernel:
%
%    img = randn(50,50,50);
%    h = ones(7,7,7);
%    tic;
%    res1 = convn(img,h);
%    toc
%    k1 = decompose_kernel(h);
%    tic;
%    res2 = img;
%    for ii=1:length(k1)
%       res2 = convn(res2,k1{ii});
%    end
%    toc
%    rms_diff = sqrt(mean((res1(:)-res2(:)).^2))

% Copyright 2010, Cris Luengo,
%   Centre for Image Analysis,
%   Swedish University of Agricultural Sciences and Uppsala University,
%   Uppsala, Sweden.
% 20 July 2010.

% Save the input kernel to compare to later on
h = h_orig;        
% Create cell array for output
n = ndims(h);
k1 = cell(1,n);
% Decompose last dimension iteratively until we've got a 2D kernel left
s = size(h);
for ii=n:-1:3
   h = reshape(h,[],s(end));       % collapse all dims except last one
   [h,k1{ii}] = decompose_2D(h);
   s = s(1:end-1);
   h = reshape(h,s);               % restore original shape with 1 fewer dim
   k1{ii} = shiftdim(k1{ii},2-ii); % the 1D kernel must be in the right shape
end
% Decompose the final 2D kernel
[k1{1},k1{2}] = decompose_2D(h);


% Reconstruct nD matrix from 1D components
kn = k1{1}*k1{2};
for ii=3:length(k1)
   s = ones(ii,1);
   s(ii) = length(k1{ii});
   kn = repmat(kn,s);
   s = size(kn);
   s(end+1:ii) = 1; % in case kn has ending singleton dimensions, which will be ignored
   s(ii) = 1;
   kn = kn.*repmat(k1{ii},s);
end

% Calculate the error we made
err = sum(abs(h_orig(:)-kn(:)));


function [h1,h2] = decompose_2D(h)
% This does the decomposition of a 2D kernel into 2 1D kernels
% More or less a direct copy-paste from the FILTER2 function
% in the standard MATLAB toolbox.
[ms,ns] = size(h);
if (ms == 1)
   h1 = 1;
   h2 = h;
elseif (ns == 1)
   h1 = h;
   h2 = 1;
else
   separable = false;
   if all(isfinite(h(:)))
      % Check rank (separability) of kernel
      [u,s,v] = svd(h);
      s = diag(s);
      tol = length(h) * eps(max(s));
      rank = sum(s > tol);   
      separable = rank==1;
   end
   if ~separable
      error('Sorry, kernel is not separable.');
   end
   h1 = u(:,1) * sqrt(s(1));
   h2 = (conj(v(:,1)) * sqrt(s(1)))';
end
