##################################################################
##################################################################
##################################################################

import numpy as np
import scipy.constants as const


#SJL 10/19
#function to calculate the residual (potentially modified) for fitting
#params: Vector of fit parameters
#p: pressure of data points
#T: temperature of data points
#met: Metal composition array for data
#sil: Silicate composition array for data
#logK_part: Partitioning coeffs to fit
#logK_part_weights: Weights for least squares fit
#flag_elem_part: array of flags (0,1) for which elements to fit
#ind_elem_part: indices of elements to fit
#ind_active_elem: indices of 'active elements' - those that are used anywhere in the fit
#flag_term: array of flags (0,1) for which terms to fit
#ind_term_fit: 2-D array of which terms to fit
#Nelem: Number of elements (global constant)
#valence: Valence/2 for each element
#lngammai0_ref: Array of reference lngammi0_ref values
#term_default: Array of any default terms (unfitted)
#flag_term_default: Which of the default terms to use
#ind_active_elem_calc: Which elements are active in the calculation inc default terms
#ind_elem_disc: Which elements (other than 0) to treat as a dissociation reaction
#flag_output_mode: output flag type
#    0*: just residual (cost) function. 
#    1: Also outputs info about which samples were used for the calc for each element (ind_elem_store, ind_samp_store),
#       array of calculated logK values (logKcalc_store),
#       Array of data logK that the calculated values were compared to (logK_part_store)
#flag_norm_Nsamp_elem
#    0*: Do not normalize my number of samples with data for partitioning for each element
#    1: Normalize by the number of samples with data for partitioning for element

def partfit(params,p,T,met,sil,logK_part,logK_part_weights,flag_elem_part,ind_elem_part,ind_active_elem,flag_term,ind_term_fit,\
            Nelem,valence,lngammai0_ref,term_default,flag_term_default,ind_active_elem_calc,\
            ind_elem_disc,flag_output_mode=0,flag_norm_Nsamp_elem=0):
    
    #total number of points to be fit
    Nsamp_fit_tot=np.shape(np.where(np.isfinite(logK_part[:,ind_elem_part])))[1]
    
    #extract the fitted parameters into a full array
    params_all=np.zeros(np.shape(flag_term))
    for i in np.arange(np.shape(ind_term_fit)[0]):
        params_all[ind_term_fit[i,0],ind_term_fit[i,1]]=params[i]
        #if compositional term, make sure its diagonal
        if (ind_term_fit[i,1]>2)&(ind_term_fit[i,1]<Nelem+3):
            params_all[ind_term_fit[i,1]-3,ind_term_fit[i,0]+3]=params[i]
        elif (ind_term_fit[i,1]>=Nelem+4):#exclude oxygen
            params_all[ind_term_fit[i,1]-3-Nelem,ind_term_fit[i,0]+3+Nelem]=params[i]
                
    #add in the default parameters
    flag_term=flag_term+flag_term_default
    params_all=params_all+flag_term_default*term_default
    
    ###############
    #find log gamma Fe in metal (from Ma 2001)
    #diagonal terms
    temp=(np.diagonal(flag_term[:,3:(Nelem+3)])*\
          np.diagonal(params_all[:,3:(Nelem+3)])*\
          (met[None,:]+np.log(1.0-met[None,:])))[0,:,:]
    
    temp=np.where(np.isfinite(met),temp,0)
    loggammaFe=np.sum(temp,axis=1)
    
    
    #do term 2 and 4 loop
    #make sure not to get the last element and compensate for no elemental terms
    if (np.size(ind_active_elem_calc)==0):
        ind_loop=ind_active_elem_calc #do nothing
    elif (ind_active_elem_calc[-1]==(Nelem-1)):
        ind_loop=ind_active_elem_calc[:-1]
    else:
        ind_loop=ind_active_elem_calc
        
    for j in ind_loop:
        for k in ind_active_elem_calc[np.where(ind_active_elem_calc>j)[0]]:
            #term 2 part 1
            temp=-params_all[j,k+3]\
                        *(met[:,j]*met[:,k]\
                        +met[:,k]*np.log(1.0-met[:,j])\
                        +met[:,j]*np.log(1.0-met[:,k]))
            #term 4 part 1
            temp+=0.5*params_all[j,k+3]\
                       *((met[:,j]*met[:,k])**2)\
                       *(1.0/(1.0-met[:,j])+1.0/(1.0-met[:,k])-1.0)
            
            loggammaFe+=np.where(np.isfinite(met[:,j])&np.isfinite(met[:,k]),temp,0)
            
    #do term 3 and 5 loop
    for i in ind_active_elem_calc:
        for k in ind_active_elem_calc:
            if k==i:
                temp=0 #do nothing
            else:
                #term 3 part 1
                temp=params_all[i,k+3]\
                           *(met[:,i]*met[:,k]\
                            +met[:,i]*np.log(1-met[:,k])\
                            -met[:,i]*met[:,k]/(1.0-met[:,i])) 
                #term 5 part 1
                temp+=-1.0*params_all[i,k+3]\
                           *((met[:,i]*met[:,k])**2)\
                           *(1.0/(1.0-met[:,i])+1.0/(1.0-met[:,k])+0.5*met[:,i]/((1.0-met[:,i])**2)-1.0)

                loggammaFe+=np.where(np.isfinite(met[:,i])&np.isfinite(met[:,k]),temp,0)
    
    
    #apply the multiplicative factors for T dependence and to account for log10
    loggammaFe=loggammaFe*1873.0/np.log(10)/T
             
    #calculate excess gibbs free energy for silicate terms
    #uses a symmetric solution model using Margules parameters (e.g., Mukhopadhyay et al., 1993, GCA)
    GXS=np.zeros(np.size(loggammaFe))
    for j in ind_active_elem_calc:
        for k in ind_active_elem_calc[np.where(ind_active_elem_calc>j)[0]]:
            GXS+=params_all[j,k+3+Nelem]*sil[:,k]*sil[:,j]
            
    #calculate the FeO activity
    loggammaFeO=-1.0*GXS
    for k in ind_active_elem_calc:
        if k!=1: #FeO activity
            loggammaFeO+=(params_all[1,k+3+Nelem]*sil[:,k])
    loggammaFeO=loggammaFeO/np.log(10)/T/const.R

    loggammaO=np.zeros(np.shape(loggammaFeO))
                      
    ###########
    #loop over each element to be fitted and calculate fit for each relevant sample
    cost=np.empty(0)
    if flag_output_mode!=0: #if need more info intialize more arrays
        ind_elem_store=np.empty(0,dtype=int)
        ind_samp_store=np.empty(0,dtype=int)
        logKcalc_store=np.empty(0)
        logK_part_store=np.empty(0)
    for i in ind_elem_part:
            
        #find which samples and terms to use
        ind_elem_samp=np.where(np.isfinite(logK_part[:,i]))[0]
        #ind_elem_term=ind_term_calc[np.where(ind_term_calc[:,0]==i)[0],1] #for metals only
        ind_elem_term=np.where(flag_term[i,:]==1)[0] #for metals only
            
        ##activity of element i in metal (Ma 2001)
        #metal activity starting with Fe
        loggammai=loggammaFe[ind_elem_samp]/1873.0*np.log(10)*T[ind_elem_samp] #need to correct back
      
        #self interaction terms in metal
        temp=-params_all[i,i+3]*np.log(1.0-met[ind_elem_samp,i])
        #loggammai+=np.where((met[ind_elem_samp,i]>0),temp,0)
        loggammai+=np.where(np.isfinite(met[ind_elem_samp,i]),temp,0)
        
        #other terms in metal
        for k in ind_elem_term[np.where(ind_elem_term>(2))[0]]:
            if k==i:
                temp=0 #do nothing for diagonal terms
            #metals
            elif (k<(3+Nelem))&(k!=4): #ignoring iron
                temp=-params_all[i,k]\
                    *(met[ind_elem_samp,k-3]\
                                           +np.log(1.0-met[ind_elem_samp,k-3])\
                                           -(met[ind_elem_samp,k-3]/(1.0-met[ind_elem_samp,i])))
                temp+=params_all[i,k]\
                    *(met[ind_elem_samp,k-3]**2)*met[ind_elem_samp,i]*\
                    ((1.0/(1.0-met[ind_elem_samp,k-3]))+(1.0/(1.0-met[ind_elem_samp,i]))\
                     +(met[ind_elem_samp,i]/(2.0*(1.0-met[ind_elem_samp,i])**2))-1.0)

                ind=np.where((np.isfinite(met[ind_elem_samp,i])&np.isfinite(met[ind_elem_samp,k-3])))[0]
                loggammai[ind]+=temp[ind]
                
        #apply multiplicative factors
        loggammai=loggammai*1873.0/np.log(10)/T[ind_elem_samp]

             
        ##activity of oxides
        #find the activity for oxide in question 
        loggammai_sil=np.zeros(np.size(ind_elem_samp))
        ind=np.where(np.isfinite(sil[ind_elem_samp,i]))[0]
        loggammai_sil[ind]=-1.0*GXS[ind_elem_samp[ind]]
        for k in ind_active_elem_calc:
            if (k!=i): #element in question activity
                ind=np.where(np.isfinite(sil[ind_elem_samp,k]))[0]
                loggammai_sil[ind]+=(params_all[i,k+3+Nelem]*sil[ind_elem_samp[ind],k])
                                  
        #apply multiplicative factor     
        loggammai_sil=loggammai_sil/np.log(10)/T[ind_elem_samp]/const.R
                    
        #add all the terms up depending on assumed reaction
        #do the front loaded terms a, b, c
        logK=params_all[i,0]+params_all[i,1]*1.0/T[ind_elem_samp]\
            +params_all[i,2]*p[ind_elem_samp]*1.0/T[ind_elem_samp]\
            +lngammai0_ref[i]*1873.0/np.log(10)/T[ind_elem_samp]
        #then composition terms
        if i==0: #for oxygen
            logK+=-loggammai-loggammaFe[ind_elem_samp]+loggammaFeO[ind_elem_samp]
            #store O for later
            loggammaO=np.nan*np.zeros(np.size(loggammaFe))
            loggammaO[ind_elem_samp]=loggammai
        elif (np.size(np.where(ind_elem_disc==i)[0])==1):
            logK+=-loggammai+loggammai_sil-valence[i]*loggammaO[ind_elem_samp]
        else:
            logK+=loggammai_sil-loggammai+valence[i]*(loggammaFe[ind_elem_samp]-loggammaFeO[ind_elem_samp])
                   
        #add the cost function to the list. Note can normalize so each element equally weighted
        cost=np.append(cost,(logK-logK_part[ind_elem_samp,i])*logK_part_weights[ind_elem_samp,i]/\
                       (1.0+flag_norm_Nsamp_elem*(np.size(ind_elem_samp)-1.0)))
        
        if flag_output_mode!=0: #extra terms to save
            ind_elem_store=np.append(ind_elem_store, i*np.ones(np.size(ind_elem_samp),dtype=int))
            ind_samp_store=np.append(ind_samp_store,ind_elem_samp)
            logKcalc_store=np.append(logKcalc_store,logK)
            logK_part_store=np.append(logK_part_store, logK_part[ind_elem_samp,i])

    #print('fish')
    #aFe=10**(loggammaFe)*met[:,1]
    #aFeO=10**(loggammaFeO)*sil[:,1]
    #print(-2.0*np.log10(aFe/aFeO))
    #print(-2.0*np.log10(met[:,1]/sil[:,1]))
            
    #retun values depending on mode
    if flag_output_mode==0:
        return cost
    elif flag_output_mode==1:
        return cost, ind_elem_store, ind_samp_store, logKcalc_store, logK_part_store
    elif flag_output_mode==2:
        return cost, ind_elem_store, ind_samp_store, logKcalc_store, logK_part_store, loggammaFe, loggammaFeO
