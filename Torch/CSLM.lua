
local CSLM={}

K=1
T=1
N=1
D=1
verbose=false

-- initialize model from U and W
function CSLM:new(U,W)
  self.__index=self
-- hyper-parameters
  self.K = W:size(1)
  self.T = W:size(2)
  self.D = W:size(3)
-- parameters
  self.U = U:view(1,self.T,self.D):contiguous():cuda()
  self.W = W:transpose(2, 3):contiguous():cuda()
  --self.W = W:contiguous():cuda()
  self.totheta= nn.MM():cuda()
  --transfo:forward(U:view(1,T,D):expand(K,T,D):contiguous():view(K*T,D):cuda()):view(K,T,T)
  self.theta = torch.DoubleTensor(self.K, self.T, self.T)
  self.theta_t = torch.DoubleTensor(self.K, self.T, self.T)
  --
  self.max_rows=torch.CudaTensor(self.K, self.T, 1)
  self.max_cols=torch.CudaTensor(self.K, 1, self.T)
-- gradients
  self.gradU = torch.CudaTensor(1, self.T, self.D) -- it's easier to compute gradU in K steps, then sum along the firs dimension
  self.gradW = torch.CudaTensor(self.K, self.D, self.T)
-- marginals
  self.mu = torch.CudaTensor(self.K+1, self.T)
  self.delta = torch.CudaTensor(self.K+1, self.T):fill(0)
  --
  self.mu2 = torch.DoubleTensor(self.K, self.T, self.T) -- not necessarily useful: strored on the GPU in mem after running md:mp(true)
  self.mem = torch.CudaTensor(self.K, self.T, self.T)
-- log-partition
  self.part = 1e3 * torch.pow(self.T, self.K+1)
  return self
end

-- initialize model from scratch
function CSLM:all_new(K, T, D)
  self.__index=self
-- hyper-parameters
  self.K = K
  self.T = T
  self.D = D
-- parameters
  self.U = torch.CudaTensor(1, self.T, self.D):uniform(-1,1)
  self.W = torch.CudaTensor(self.K, self.D, self.T):uniform(-1,1)
  self.totheta= nn.MM():cuda()
  --transfo:forward(U:view(1,T,D):expand(K,T,D):contiguous():view(K*T,D):cuda()):view(K,T,T)
  self.theta = torch.DoubleTensor(self.K, self.T, self.T)
  self.theta_t = torch.DoubleTensor(self.K, self.T, self.T)
  --
  self.max_rows=torch.CudaTensor(self.K, self.T, 1)
  self.max_cols=torch.CudaTensor(self.K, 1, self.T)
-- gradients
  self.gradU = torch.CudaTensor(1, self.T, self.D) -- it's easier to compute gradU in K steps, then sum along the firs dimension
  self.gradW = torch.CudaTensor(self.K, self.D, self.T)
-- marginals
  self.mu = torch.CudaTensor(self.K+1, self.T)
  self.delta = torch.CudaTensor(self.K+1, self.T):fill(0)
  --
  self.mu2 = torch.DoubleTensor(self.K, self.T, self.T) -- not necessarily useful: strored on the GPU in mem after running md:mp(true)
  self.mem = torch.CudaTensor(self.K, self.T, self.T)
-- log-partition
  self.part = 1e3 * torch.pow(self.T, self.K+1)
  return self
end

-- makes theta, keeps a version on the CPU and 3 on the GPU (precomputes the exponential twice to avoid overflow)
-- requires a GPU with at least 4GB for K=2
function CSLM:make_theta()
  self.thetap = self.totheta:forward({self.U:expand(self.K, self.T, self.D), self.W}):mul(self.K+1)
  self.theta = self.thetap:double()
--print(self.theta[1][2][3])
  self.max_rows=self.theta:max(3):cuda()
  self.max_cols=self.theta:max(2):cuda()
  pretheta=nil
  collectgarbage()
  self.phi_rows=self.thetap:clone():add(-1, self.max_rows:expand(self.K, self.T, self.T) ):exp()
  collectgarbage()
  self.phi_cols=self.thetap:clone():add(-1, self.max_cols:expand(self.K, self.T, self.T) ):exp()
  collectgarbage()
end

-- ensures that the maximum norm of an embedding is maxn (default 2)
function CSLM:normal(maxn)
  local maxn=maxn or 2
  self.U:view(self.T, self.D):renorm(2,1,maxn):view(1, self.T, self.D)
  self.W=self.W:double():permute(1,3,2):contiguous()
  self.W:view(self.K*self.T, self.D):renorm(2,1,maxn):view(self.K, self.T, self.D)
  self.W=self.W:permute(1,3,2):contiguous():cuda()
end

-- reinitializes embeddings at random with a max norm of 2, does NOT call make_theta
function CSLM:random_init()
  self.U:uniform(-1,1)
  self.W:uniform(-1,1)
  self:normal(2)
  self.delta:fill(0)
end

-- message passing in log space to avoid overflow for bigger GPUs
function CSLM:big_mp(final)
  t = sys.clock()
  sys.tic()
  local final = final or false
  local phi=self.mem:copy(self.phi_rows) --phi_rows=exp(theta-max_rows)
--
  local max_delta=self.delta:max(2)
  local exp_delta = self.delta:clone():add(-1, max_delta:expand(self.K+1, self.T)):exp()
--
  local log_psi = torch.CudaTensor(self.K+1,self.T):fill(0)
  local psi = torch.CudaTensor(self.K+1,self.T)
  local messages_up = torch.CudaTensor(self.K,self.T)
  local messages_down = torch.CudaTensor(self.K,self.T)
--going up
  local pre_exp=exp_delta:sub(1,self.K):view(self.K, 1, self.T):expand(self.K, self.T, self.T)
  phi:cmul(pre_exp) -- phi[k][i][j]=exp(theta[k][i][j] + delta[k][j] - max_j(theta[k][i][j]) - max_j(delta[k][j]) )
  t=sys.toc()
  --if verbose then print("time f",t) end
--
  if verbose then print('phimin',phi:min(),'phisummin',phi:sum(3):min()) end
  messages_up = phi:sum(3):log()
  messages_up:add( max_delta:sub(1, self.K):expand(self.K, self.T) )
  messages_up:add( self.max_rows:view(self.K, self.T) )
--
  log_psi[self.K+1] = messages_up:sum(1)
  log_psi[self.K+1]:add( self.delta[self.K+1] )
  log_psi:sub(1, self.K):add(-1, messages_up)
  log_psi:sub(1, self.K):add(log_psi[self.K+1]:view(1, self.T):expand(self.K, self.T))
  if verbose then print('exp_delta',exp_delta:sum(),'phisum',phi:sum(),'logpsisum',log_psi:sum()) end
--
  local max_psi = log_psi:max(2)
  log_psi:add(-1, max_psi:expand(self.K + 1, self.T))
  psi:exp(log_psi)
--
  local log_part = torch.log( psi[self.K+1]:sum() ) + max_psi[self.K+1][1]
  log_psi:add(max_psi:expand(self.K + 1, self.T))
  self.mu[self.K+1]=log_psi[self.K+1]:double()
  self.mu[self.K+1]:add( -log_part )
  self.mu[self.K+1]:exp()
  t=sys.toc()
  --if verbose then print("time g",t) end
--pairwise
  if  final then   
    self.mem:copy(self.thetap)
    self.mem:add( log_psi:sub(1, self.K):view(self.K, self.T, 1):expand(self.K, self.T, self.T) )
    self.mem:add( self.delta:sub(1, self.K):view(self.K, 1, self.T):expand(self.K, self.T, self.T) )
    self.mem:add( -1*log_part )
    self.mu2 = self.mem:exp():double()
  else
  --going down
    phi=self.mem:copy(self.phi_cols) --phi_cols=exp(theta-max_cols)
    pre_exp = psi:sub(1,self.K):view(self.K, self.T, 1):expand(self.K, self.T, self.T)
    phi:cmul(pre_exp)
    --print('maxphi',phi:max(),'maxpre_exp',pre_exp:max())
    t=sys.toc()
    --if verbose then print("time j",t) end
    messages_down = phi:sum(2):view(self.K, self.T)
--if verbose then print('phisum',phi:sum(),'psisum',psi:sum(),'logpsisum',log_psi:sum()) end
--if verbose then print('exp-md[2][1]',messages_down[2][1],'log-part',log_part) end
--print('maxmes',messages_down:max(),'minmes',messages_down:min(), 'minpreexp', pre_exp:min())
    messages_down:log()
--if verbose then print('log-md[2][1]',messages_down[2][1],'log-part',log_part) end
    messages_down:add(max_psi:sub(1, self.K):expand(self.K, self.T))
    messages_down:add(self.max_cols:view(self.K, self.T))
    messages_down:add(self.delta:sub(1, self.K))
--if verbose then print('log-md[2][1]',messages_down[2][1],'log-part',log_part) end
--print('maxmes',messages_down:max(),'minmes',messages_down:min(),'logpart',log_part)
  --
    self.mu:sub(1, self.K):exp( messages_down:add( -1*log_part ) )
  end
  --collectgarbage()
  self.part = log_part
  t=sys.toc()
  --if verbose then print("time l",t) end
  return log_part
end

-- simple dual decomposition
function CSLM:dd(rate)
  local rate=rate or 10
  local grad_delta = torch.CudaTensor(self.K+1, self.T):fill(0)
  local oldpart=math.huge
  collectgarbage()
  local function hasnan()
    local allreal = (self.mu:sum()==self.mu:sum()) and (self.part==self.part)
    return not allreal
  end
  for i=1, 50 do
    oldpart=self.part  
    md:big_mp()
    --if i==3 then print(self.part, self.delta:norm(), self.delta:sum(), grad_delta:norm(), grad_delta:sum()) end
    --
    if self.part > oldpart or hasnan() then
      if verbose then print("back") end
      self.delta:add(-rate, grad_delta)
      rate = rate/2
      if not i==50 then self.part = oldpart end
    else
      grad_delta=grad_delta:fill(0)
      for l=1, K do
        grad_delta[l]=grad_delta[l]:add(self.mu[self.K + 1], -1, self.mu[l])
        grad_delta[self.K + 1]=grad_delta[self.K + 1]:add(-1, grad_delta[l])
      end
 --     if verbose then print("mu[1][1]", md.mu[1][1], "mu[2][1]", md.mu[2][1], "mu[3][1]", md.mu[3][1])
 --       print("delta[1][1]", md.delta[1][1], "delta[2][1]", md.delta[2][1],"delta[3][1]", md.delta[3][1]) end
    end
    if verbose then print(self.part) end
    self.delta:add(rate, grad_delta)
  end
  --print(self.part)
end

-- optim-based dual decomposition (currently using LBFGS)
function CSLM:dd_opt(rate, conf)
  local grad_delta = torch.CudaTensor(self.K+1, self.T):fill(0)
  local function hasnan()
    local allreal = (self.mu:sum()==self.mu:sum()) and (self.part==self.part)
    return not allreal
  end
  
  local delta_opt=torch.DoubleTensor((self.K+1)*self.T):copy(self.delta:view((self.K+1)*self.T))
  local grad_delta_opt=torch.DoubleTensor((self.K+1)*self.T)
 
  local function feval(x)
    self.delta = x:view(self.K+1, self.T):cuda()
    local obj=self:big_mp()
    grad_delta=grad_delta:fill(0)
    for l=1, K do
      grad_delta[l]=grad_delta[l]:add(self.mu[self.K + 1], -1, self.mu[l])
      grad_delta[self.K + 1]=grad_delta[self.K + 1]:add(-1, grad_delta[l])
    end
    grad_delta_opt:copy(grad_delta:view((self.K+1)*self.T)):mul(-1)
    --print('log_part',obj,'gradient-norm',grad_delta:norm())
    return obj, grad_delta_opt
  end
  
  conf = conf or {}
  config={}
  config.maxIter = conf.maxIter or 50
  config.maxEval = conf.maxEval or 100
  config.tolFun = conf.tolFun or 1e-5
  config.tolX = conf.tolX or 1e-5
  config.lineSearch = conf.lineSearch or optim.lswolfe
  config.verbose=false
  optim.lbfgs(feval, delta_opt, config)
  return grad_delta:norm()
end

-- back-propagates the moments in theta to U,W
function CSLM:make_grad(data, reg)
  reg=reg or 0
  --print('mu2sum',self.mu2:sum(),'mu1sum1',self.mu[1]:sum(),'mu1sum2',self.mu[2]:sum(),'mu1sum3',self.mu[3]:sum())
  local grad=self.mem:copy(torch.DoubleTensor():add(data, -1, self.mu2))
  local gradPar=self.totheta:backward({self.U:expand(self.K, self.T, self.D), self.W}, grad)
  self.gradU=gradPar[1]:sum(1):add(-reg,self.U)
  self.gradW=gradPar[2]:add(-reg,self.W)
end

-- gradient_step isn't used with the optim framework
function CSLM:gradient_step(data, rate, norm, reg)
  print('|U[1]|   ',self.U[1][1]:norm(),'|U[2]|   ',self.U[1][2]:norm())
  print('|W[1][1]|   ',self.W:norm(2,2)[1][1][1],'|W[2][1]|   ',self.W:norm(2,2)[2][1][1])
  rate = rate or 10
  norm = norm or false
  reg = reg or 0
  md.delta:fill(0)
  self:dd(10,true)
  local obj=self:objective(data)
  print('Objective', obj, 'Regularized',obj - reg * (self.U:norm() + self.W:norm()), 'deltasum', self.delta:sum())
  self:big_mp(true)
  self:make_grad(data)
  ---option 1: renormalize after each gradient step
  if norm then
    self.U:add(rate, self.gradU)
    self.W:add(rate, self.gradW)
    self:normal()
  ---option 2: L2 regularization
  else
    self.gradU:add(-reg, self.U)
    self.gradW:add(-reg, self.W)
    self.U:add(rate, self.gradU)
    self.W:add(rate, self.gradW)
  end
  print('|gradU|   ',self.gradU:norm(),'|gradW|   ',self.gradW:norm())
  self:make_theta()
end

-------- some functions for the optimisation packages
-- computes -objective (because the optim package minimizes)
function CSLM:objective(data, reg, pre)
  local pre = pre or -1
  local score = torch.DoubleTensor(self.K, self.T, self.T):cmul(data, self.theta)
  local sc=score:sum()
  local nm=self.U:norm()+self.W:norm()
  --print('Score', sc, 'Partition', self.part, 'norm-U-W', nm, 'dd-prec', pre, 'Objective', (sc - self.part) / (K+1))
  print('Score', sc, 'Partition', self.part, 'dd-prec', pre, 'Objective', (sc - self.part) / (K+1), 'Regularized', (sc - self.part) / (K+1) - reg * nm)
  return -(sc - self.part) / (K+1) + reg * nm
end

-- flattens the parameters and gradients (also multiplies the gradients by -1 to fit within the minimization framework)
function CSLM:getParameters()
  self.parameters=torch.cat(self.U:view(self.T*self.D):double(), self.W:view(self.K*self.T*self.D):double())
  self.gradParameters=torch.cat(self.gradU:view(self.T*self.D):double(), self.gradW:view(self.K*self.T*self.D):double())
  self.gradParameters:mul(-1)
end

-- optimization using LBFGS
function CSLM:optimize(data, conf)
  self:getParameters()
  local pre
  local conf = conf or {}
  local reg=conf.reg or 0
  --
  local function feval(x)
    self.U=x:sub(1,self.T*self.D):view(1, self.T, self.D):cuda()
    self.W=x:sub(self.T*self.D+1, (self.K+1)*self.T*self.D):view(self.K, self.D, self.T):cuda()
    self:make_theta()
    pre=self:dd_opt()
    local obj=self:objective(data, reg, pre)
    self:big_mp(true)
    self:make_grad(data,reg)
    self:getParameters()
    return obj, self.gradParameters
  end
  --
  config={}
  config.maxIter = conf.maxIter or 20
  config.maxEval = conf.maxEval or 50
  config.tolFun = conf.tolFun or 1e-9
  config.tolX = conf.tolX or 1e-15
  config.lineSearch = conf.lineSearch or optim.lswolfe
  config.verbose=true
  optim.lbfgs(feval, self.parameters, config)
end
