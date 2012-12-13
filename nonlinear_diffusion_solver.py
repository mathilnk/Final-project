from dolfin import  *
import time
import numpy as np
import scitools.std as sci
from mayavi import mlab
import glob
import os
class DiffProblem:
    def __init__(self, rho, alpha, f, I):
        self.rho = rho
        self.alpha = alpha # function
        self.f = f # streng
        self.I = I #streng

class Solve:
    def __init__(self,problem,dt,degree,divisions, N=100):
        self.dt = dt
        self.problem = problem
        self.N = N
        self.degree = degree #degree to the lagrange basis functions
        self.divisions = divisions #list of integers
        self.dim = len(divisions)
        self.nx = divisions[0]
        if len(divisions)>=2:
            self.ny = divisions[1]
        else:
            self.ny = 0
        


    def convergence(self, u_e_text):
        """
        u_e is a string giving the exact solution.
        """
        problem = self.problem
        T = 3
        f_text = problem.f;alpha = problem.alpha; rho = problem.rho; I_text = problem.I;
        divs = [20,30,40] #number of elements along the x-axis
        E = np.zeros(len(divs),tuple)
        counter = 0
        for d in divs:
            dxx = 1.0/d
            dt = (dxx)**2
            N = int(T/dt)
            V,m = self.make_mesh([d for i in range(self.degree)])       
            u = TrialFunction(V)
            v = TestFunction(V)
            I = Expression(I_text)
            u_k = interpolate(I,V)
            a = (u*v + dt/rho*alpha(u_k)*inner(nabla_grad(u),nabla_grad(v)))*dx
            u = Function(V)
            En = np.zeros(N)
            for n in range(N):
                t = dt*(n+1)
                f = Expression(f_text, rho=rho, t=t, dt = dt)
                L = (u_k*v + dt*f*v/rho)*dx
                solve(a==L,u)
                u_k.assign(u)
                t = dt*(n+1)
                expr = Expression(u_e_text,t=t, pi=np.pi)
                u_e = interpolate(expr,V)
                e = abs(u_e.vector().array() - u.vector().array())
                En[n] =np.sqrt(np.sum(e**2)/u.vector().array().size)
            E[counter] = [np.sum(En)/(len(En)*dt),dt]
            counter+=1
        print E
            
                    
                    
                    

    def make_mesh(self, div=None):
        problem,N, degree= self.problem, self.N, self.degree
        if div==None:
            divisions = self.divisions
        else:
            divisions = div
        d = len(divisions)
        domain_type = [UnitInterval, UnitSquare, UnitCube]
        mesh = domain_type[d-1](*divisions)
        V = FunctionSpace(mesh, 'Lagrange', degree)
        return V, mesh

    def make_numpy(self, u, nx,ny):
        numVec2D = u.vector().array()
        m = np.zeros((ny+1,nx+1))
        first = -(nx+1)
        last = None
        for i in range(ny+1):
            m[i] = numVec2D[first:last]
            last = first
            first = first-(nx+1)
        return m
    def mayavi_plot(self,nx,ny,m, filename_whole):
        mlab.options.offscreen=True
        X,Y=sci.meshgrid(np.linspace(0,1,nx+1),np.linspace(0,1,ny+1))
        mlabmesh = mlab.mesh(X,Y,m)
        mlab.savefig(filename_whole)
        mlab.clf()
        
    def plot1D(self,u,nx,ny,exact,mesh,ax,t=0, n=0):
        numVec = self.make_numpy(u,nx,ny)
        x = np.linspace(0,1,100)
        if exact!=None:
            sci.plot(x, exact(x,t),mesh.coordinates(),numVec, axis=ax, legend=['Exact solution','Numerically found'], savefig='tmp%.4d'%n)
        else:
            sci.plot(mesh.coordinates(),numVec, axis=ax,legend='Numerically found', savefig='tmp%04d'%n)
        

            
    def plot2D(self,u,nx,ny,navn,n):
        m=self.make_numpy(u,nx,ny)
        self.mayavi_plot(nx,ny,m,"%s%.4d.png"%(navn,n+1))
        
    def plotError(self, u, exact, nx, ny,mesh,t, navn,n=0):
        u_array= self.make_numpy(u,nx,ny)
        u_e_array = exact(mesh.coordinates(), t)
        e = abs(u_e_array[:,0] - u_array[0])
       
        if ny==0:
            sci.plot(mesh.coordinates(),e, title="Error %s"%navn, xlabel="x", ylabel="error: |u-u_e|", savefig="ErrorPlot%04d.png"%(n+1))#, axis=[0,1,0,0.00004])
            #time.sleep(2)
            
        else:
            self.mayavi_plot(nx,ny,e,"ErrorPlot%4d.png"%(n+1))
        
        
    def solve3(self, exact=None, s=0, ax=[], navn="tmp"):
        if self.dim>2:
            return None
        problem,dt, N  = self.problem, self.dt, self.N
        f_text = problem.f;alpha = problem.alpha; rho = problem.rho; I = problem.I;
        nx,ny = self.nx, self.ny
        V,mesh = self.make_mesh()
        u = TrialFunction(V)
        v = TestFunction(V)
        I = Expression(I, s=s)# can take one parameter with the name s
        u_k = interpolate(I,V)
        a = (u*v + dt/rho*alpha(u_k)*inner(nabla_grad(u),nabla_grad(v)))*dx
        u = Function(V)
        
        if self.dim==2:
            mlab.figure(1)
            self.plot2D(u_k,nx,ny,navn,-1)
         
        if self.dim==1:
            sci.figure(1)
            self.plot1D(u_k,nx,ny,exact,mesh,ax)
            
        for n in range(N):
            time.sleep(0.2)
            t = (n+1)*dt
            f = Expression(f_text, rho=rho, t=t, dt = dt)
            L = (u_k*v + dt*f*v/rho)*dx
            solve(a==L,u)
            numVec  = self.make_numpy(u,nx,ny)
            
            if self.dim==2:
                mlab.figure(1)
                self.plot2D(u,nx,ny,navn,n)
            if self.dim == 1:
                sci.figure(1)
                self.plot1D(u,nx,ny,exact,mesh,ax,t=t,n=n+1)
            if exact != None:
                if ny==0:
                    sci.figure(2)
                else:
                    mlab.figure(2)
                self.plotError(u,exact,nx,ny,mesh,t, navn, n=n)
            u_k.assign(u)
            
        if self.dim==2:
            self.makemovie(navn)
        r = raw_input('Done solving!')


            
            
    def makemovie(self, navn):
        sci.movie("%s*.png"%navn, encoder='convert', fps=10, output_file='%s_movie.gif'%navn)
        for i in glob.glob("%s*.png"%navn):
            os.remove(i)
    
            
            
        

class Manufactured:
    def __init__(self):
        self.I = "0"
        #self.f = "-rho*x[0]*x[0]*x[0]/3.0 + rho*x[0]*x[0]/2.0 + (8.0/9)*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0]*x[0]*x[0] - 28/9.0*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0]*x[0]/ + 7/2.0*t*t*t*x[0]*x[0]*x[0]*x[0]*x[0] - (5/4.0)*t*t*t*x[0]*x[0]*x[0]*x[0] + 2*t*x[0] - t"
        self.f = '-rho*x[0]*x[0]*x[0]/3. + rho*x[0]*x[0]/2. + pow(t,3)*pow(x[0],4)*(pow(x[0],3)*8./9. - \
        28.*pow(x[0], 2)/9. + 7.*pow(x[0],1)/2. - 5./4.) + 2.*t*x[0] - t'
        self.rho = 1
        self.dt = 0.01
        self.degree = 1
        self.division = [20]
        self.u_sol_text = "t*pow(x[0],2)*(0.5 - x[0]/3)"
    def u_sol(self,x,t):
        return t*x**2*(0.5-x/3)

    def alpha(self,u):
        return 1+u**2
    def make(self):
        prob = DiffProblem(self.rho, self.alpha, self.f, self.I)
        self.prob = prob
        s = Solve(prob, self.dt, self.degree, self.division)
        self.s = s
        
    def make_and_solve(self):
        self.make()
        self.s.solve3(self.u_sol, ax=[0,1,0,0.2], navn="not_so_good_f")

    def make_and_solve_2(self):
        dt = self.dt
        self.f = "rho*pow(x[0],2)*(-2*x[0]+3)/6.0 - (-12*t*x[0] + 3*t*(-2*x[0]+3))*(pow(x[0],4)*pow((-dt+t),2)*pow(-2*x[0]+3,2)+36)/324.0 - (-6*t*pow(x[0],2) + 6*t*x[0]*(-2*x[0] + 3))*(36*pow(x[0],4)*pow(-dt+t,2)*(2*x[0]-3) + 36*pow(x[0],3)*pow(-dt+t,2)*pow((-2*x[0] + 3),2))/5832.0"
        self.make()
        self.s.solve3(self.u_sol,ax=[0,1,0,0.2], navn="better_f")

class Gauss:
    def __init__(self, b, s, degree, dim):
        self.b = b
        self.s = s
        self.degree = degree
        I_1 = "exp(-1.0/(2*pow(s,2))*(pow(x[0],2)))"
        I_2 = "exp(-1.0/(2*pow(s,2))*(pow(x[0],2) + pow(x[1],2)))"
        I_3 = "exp(-1.0/(2*pow(s,2))*(pow(x[0],2) + pow(x[1],2) + pow(x[2],2)))"
        I_list = [I_1,I_2,I_3]
        self.I  = I_list[dim-1]
        self.f = "0"
        self.alpha = lambda u: 1 + b*u**2
        
        self.division = [30 for i in range(dim)]
        self.rho = 1
        self.dt = 0.01

    def make(self):
        prob = DiffProblem(self.rho, self.alpha, self.f, self.I)
        self.prob = prob
        solv = Solve(prob, self.dt, self.degree, self.division,N=50)
        self.solv = solv
    def make_and_solve(self):
        self.make();
        self.solv.solve3(s = self.s, ax=[0,1,0,1], navn="Gauss")#axis is for the 1D-problem
        
        
        
class First_Verification:
    def __init__(self):
        self.p = DiffProblem(1,lambda u: 1, "0", "cos(pi*x[0])")
        s = Solve(self.p,0.03,1, [30],N = 70)
        #s.convergence("exp(-pi*pi*t)*cos(pi*x[0])")
        s.solve3(lambda x,t: np.exp(-np.pi**2*t)*np.cos(np.pi*x),ax=[0,1,-1,1])
        
        
    
        
            
      
        
if __name__ == '__main__':
    #p = DiffProblem(1,lambda u: 1, "0", "cos(pi*x[0])")
    #s = Solve(p,0.01,1, [30],N = 70)
    #s.convergence("exp(-pi*pi*t)*cos(pi*x[0])")
    #s.solve3(lambda x,t: np.exp(-np.pi**2*t)*np.cos(np.pi*x),ax=[0,1,-1,1])
    #manu = Manufactured()
    #manu.make_and_solve_2()
    #First_Verification()
    gauss = Gauss(1,0.5,1,2)
    gauss.make_and_solve()



#to run for 2d: 'xvfb-run --server-args="-screen 0 1024x768x24" python nonlinear_diffusion_solver.py' 

"""
E/h:                     h:
[[0.065502086669548307, 0.25]
[0.087447037443025996, 0.1111111111111111]
 [0.099476001443932788, 0.04000000000000001]
 [0.10099006738394452, 0.010000000000000002]
 [0.10072715750493436, 0.00591715976331361]
 [0.10024578928554419, 0.0034602076124567475]
 [0.10013262516723655, 0.0025000000000000005]
 [0.099572827115609219, 0.0011111111111111111]]

 [[0.099303925949595487, 0.0006250000000000001]
 [0.099093053262512262, 0.0004]
 [0.098959422723820634, 0.0002777777777777778]]


"""


"""
convergence test for g)
[[0.031755123265131774, 0.25]
[0.023687466325278839, 0.1111111111111111]
[0.021283586788617487, 0.0625]
[0.018922639481795962, 0.010000000000000002]]
[[0.018961356013537989, 0.0025000000000000005]
[0.019030467166668275, 0.0011111111111111111]
[0.019028313478088671, 0.0006250000000000001]]
"""
