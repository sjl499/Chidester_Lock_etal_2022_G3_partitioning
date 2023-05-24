#SJL 12/19
#read and write binary files for fits

import numpy as np
import struct

def write_fit_binary(binary_file, filename, Nelem, elem,elem_met, elem_sil, valence, ind_elem_disc, \
              Nelem_part, flag_elem_part,ind_elem_part,flag_term,ind_term_fit,\
             flag_LS, flag_MC, flag_norm_Nsamp_elem,flag_calc_weights,Ntest, NMC, sigma_omit,\
             lngammai0_ref, term_default,flag_term_default,\
                     Nsamp, study_names, sample_names,\
             p, T, met, sil, logK_part, logK_part_err, \
             ind_active_elem, ind_active_elem_calc, \
             fits, cov, corr):
    
    #open the file
    f=open(binary_file, 'wb')
    
    #go through and write out the various elements
    f.write(struct.pack('500s',filename.ljust(500).encode()))
    f.write(struct.pack('i',Nelem))
    
    for i in np.arange(Nelem):
        f.write(struct.pack('50s',elem[i].ljust(50).encode()))
        
    for i in np.arange(Nelem):
        f.write(struct.pack('50s',elem_met[i].ljust(50).encode()))
        
    for i in np.arange(Nelem):
        f.write(struct.pack('50s',elem_sil[i].ljust(50).encode()))
        
    f.write(struct.pack(str(Nelem)+'d',*valence))
    
    f.write(struct.pack('i',np.size(ind_elem_disc))) #print out the number of disc elements for read in
    if np.size(ind_elem_disc)!=0:
        f.write(struct.pack(str(np.size(ind_elem_disc))+'i',*np.asarray(ind_elem_disc)))
        
    f.write(struct.pack('i',Nelem_part))
    f.write(struct.pack(str(Nelem)+'i',*flag_elem_part))
    f.write(struct.pack(str(Nelem_part)+'i',*ind_elem_part))
    
    for i in np.arange(Nelem):
        f.write(struct.pack(str(3+2*Nelem)+'i',*flag_term[i,:]))
        
    Nterm_fit=np.shape(ind_term_fit)[0]
    f.write(struct.pack('i',Nterm_fit))
    f.write(struct.pack(str(Nterm_fit)+'i',*ind_term_fit[:,0]))
    f.write(struct.pack(str(Nterm_fit)+'i',*ind_term_fit[:,1]))
    
    f.write(struct.pack('i',flag_LS))
    f.write(struct.pack('i',flag_MC))
    f.write(struct.pack('i',flag_norm_Nsamp_elem))
    f.write(struct.pack('i',flag_calc_weights))
    f.write(struct.pack('i',Ntest))
    f.write(struct.pack('i',NMC))
    f.write(struct.pack('i',sigma_omit))
    
    f.write(struct.pack(str(Nelem)+'d',*lngammai0_ref))
    for i in np.arange(Nelem):
        f.write(struct.pack(str(3+2*Nelem)+'d',*term_default[i,:]))
    for i in np.arange(Nelem):
        f.write(struct.pack(str(3+2*Nelem)+'i',*flag_term_default[i,:]))
        
    f.write(struct.pack('i',Nsamp))
        
    for i in np.arange(Nsamp):
        f.write(struct.pack('500s',study_names[i].ljust(500).encode()))
    for i in np.arange(Nsamp):
        f.write(struct.pack('500s',sample_names[i].ljust(500).encode()))
    f.write(struct.pack(str(Nsamp)+'d',*p))
    f.write(struct.pack(str(Nsamp)+'d',*T))
    
    for i in np.arange(Nelem):
        f.write(struct.pack(str(Nsamp)+'d',*met[:,i]))
        
    for i in np.arange(Nelem):
        f.write(struct.pack(str(Nsamp)+'d',*sil[:,i]))
        
    for i in np.arange(Nelem):
        f.write(struct.pack(str(Nsamp)+'d',*logK_part[:,i]))
        
    for i in np.arange(Nelem):
        f.write(struct.pack(str(Nsamp)+'d',*logK_part_err[:,i]))
       
    Nactive_elem=np.size(ind_active_elem)
    f.write(struct.pack('i',Nactive_elem))
    f.write(struct.pack(str(Nactive_elem)+'i',*ind_active_elem))
    
    Nactive_elem_calc=np.size(ind_active_elem_calc)
    f.write(struct.pack('i',Nactive_elem_calc))
    f.write(struct.pack(str(Nactive_elem_calc)+'i',*ind_active_elem_calc))
    
    f.write(struct.pack(str(Nterm_fit)+'d',*fits))
    
    for i in np.arange(Nterm_fit):
        f.write(struct.pack(str(Nterm_fit)+'d',*cov[i,:]))
    for i in np.arange(Nterm_fit):
        f.write(struct.pack(str(Nterm_fit)+'d',*corr[i,:]))
    
    
    return
    
    
def read_fit_binary(binary_file):
    f=open(binary_file, 'rb')
    
    filename=f.read(500).rstrip().decode('ascii')
    Nelem=struct.unpack('i', f.read(4))[0]
    
    elem=[None]*Nelem
    for i in np.arange(Nelem):
        elem[i]=f.read(50).rstrip().decode('ascii')
    elem=np.asarray(elem)
    
    elem_met=[None]*Nelem
    for i in np.arange(Nelem):
        elem_met[i]=f.read(50).rstrip().decode('ascii')
    elem_met=np.asarray(elem_met)
    
    elem_sil=[None]*Nelem
    for i in np.arange(Nelem):
        elem_sil[i]=f.read(50).rstrip().decode('ascii')
    elem_sil=np.asarray(elem_sil)
    
    valence=np.asarray(struct.unpack(str(Nelem)+'d',f.read(Nelem*8)))
    
    temp=struct.unpack('i', f.read(4))[0]
    if temp==0:
        ind_elem_disc=np.asarray([])
    else:
        ind_elem_disc=np.asarray(struct.unpack(str(temp)+'i',f.read(temp*4)))
        
    Nelem_part=struct.unpack('i', f.read(4))[0]
    flag_elem_part=np.asarray(struct.unpack(str(Nelem)+'i',f.read(Nelem*4)))
    ind_elem_part=np.asarray(struct.unpack(str(Nelem_part)+'i',f.read(Nelem_part*4)))
    
    flag_term=np.zeros((Nelem,3+2*Nelem),dtype=int)
    for i in np.arange(Nelem):
        flag_term[i,:]=np.asarray(struct.unpack(str(3+2*Nelem)+'i',f.read((3+2*Nelem)*4)))
        
    Nterm_fit=struct.unpack('i', f.read(4))[0]
    
    ind_term_fit=np.zeros((Nterm_fit,2),dtype=int)
    ind_term_fit[:,0]=np.asarray(struct.unpack(str(Nterm_fit)+'i',f.read(Nterm_fit*4)))
    ind_term_fit[:,1]=np.asarray(struct.unpack(str(Nterm_fit)+'i',f.read(Nterm_fit*4)))
    
    flag_LS=struct.unpack('i', f.read(4))[0]
    flag_MC=struct.unpack('i', f.read(4))[0]
    flag_norm_Nsamp_elem=struct.unpack('i', f.read(4))[0]
    flag_calc_weights=struct.unpack('i', f.read(4))[0]
    Ntest=struct.unpack('i', f.read(4))[0]
    NMC=struct.unpack('i', f.read(4))[0]
    sigma_omit=struct.unpack('i', f.read(4))[0]
    
    lngammai0_ref=np.asarray(struct.unpack(str(Nelem)+'d',f.read(Nelem*8)))
    
    term_default=np.zeros((Nelem,3+2*Nelem))
    for i in np.arange(Nelem):
        term_default[i,:]=np.asarray(struct.unpack(str(3+2*Nelem)+'d',f.read((3+2*Nelem)*8)))
        
    flag_term_default=np.zeros((Nelem,3+2*Nelem),dtype=int)
    for i in np.arange(Nelem):
        flag_term_default[i,:]=np.asarray(struct.unpack(str(3+2*Nelem)+'i',f.read((3+2*Nelem)*4)))
        
    Nsamp=struct.unpack('i', f.read(4))[0]
    
    study_names=[None]*Nsamp
    for i in np.arange(Nsamp):
        study_names[i]=f.read(500).rstrip().decode('ascii')
    study_names=np.asarray(study_names)

    sample_names=[None]*Nsamp
    for i in np.arange(Nsamp):
        sample_names[i]=f.read(500).rstrip().decode('ascii')
    sample_names=np.asarray(sample_names)
    
    p=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))
    T=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))

    met=np.zeros((Nsamp,Nelem))
    for i in np.arange(Nelem):
        met[:,i]=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))
        
    sil=np.zeros((Nsamp,Nelem))
    for i in np.arange(Nelem):
        sil[:,i]=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))
        
    logK_part=np.zeros((Nsamp,Nelem))
    for i in np.arange(Nelem):
        logK_part[:,i]=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))
        
    logK_part_err=np.zeros((Nsamp,Nelem))
    for i in np.arange(Nelem):
        logK_part_err[:,i]=np.asarray(struct.unpack(str(Nsamp)+'d',f.read(Nsamp*8)))
        
    Nactive_elem=struct.unpack('i', f.read(4))[0]
    ind_active_elem=np.asarray(struct.unpack(str(Nactive_elem)+'i',f.read(Nactive_elem*4))).tolist()
    
    Nactive_elem_calc=struct.unpack('i', f.read(4))[0]
    ind_active_elem_calc=np.asarray(struct.unpack(str(Nactive_elem_calc)+'i',f.read(Nactive_elem_calc*4))).tolist()
    
    fits=np.asarray(struct.unpack(str(Nterm_fit)+'d',f.read(Nterm_fit*8)))
    
    cov=np.zeros((Nterm_fit,Nterm_fit))
    for i in np.arange(Nterm_fit):
        cov[i,:]=np.asarray(struct.unpack(str(Nterm_fit)+'d',f.read(Nterm_fit*8)))
        
    corr=np.zeros((Nterm_fit,Nterm_fit))
    for i in np.arange(Nterm_fit):
        corr[i,:]=np.asarray(struct.unpack(str(Nterm_fit)+'d',f.read(Nterm_fit*8)))
    
    return filename, Nelem, elem,elem_met, elem_sil, valence, ind_elem_disc, \
              Nelem_part, flag_elem_part,ind_elem_part,flag_term,ind_term_fit,\
             flag_LS, flag_MC, flag_norm_Nsamp_elem,flag_calc_weights,Ntest, NMC, sigma_omit,\
             lngammai0_ref, term_default,flag_term_default,\
                     Nsamp, study_names, sample_names,\
             p, T, met, sil, logK_part, logK_part_err, \
             ind_active_elem, ind_active_elem_calc, \
             fits, cov, corr
    
       
