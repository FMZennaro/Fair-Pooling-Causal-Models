
import numpy as np
import tensorflow as tf
import edward as ed

import scipy.stats as stats

import matplotlib.pyplot as plt

nsamples = 10000

def MA_standard_eval(model,applicant):
    return model.eval(feed_dict={v_a_age: applicant['age'],
                             u_a_gnd: applicant['gnd'],
                             u_a_dpt: applicant['dpt'],
                             u_a_mrk: applicant['mrk'],
                             u_a_job: applicant['job'],
                             u_a_cvr: applicant['cvr']})
    
def MA_fair_eval(model,applicant,niter):
    return [model.eval(feed_dict={u_a_dpt: applicant['dpt'],
                             u_a_mrk: applicant['mrk'],
                             u_a_cvr: applicant['cvr']}) for _ in range(niter)]    
    
    
def MB_standard_eval(model,applicant):
    return model.eval(feed_dict={v_b_age: applicant['age'],
                             u_b_gnd: applicant['gnd'],
                             u_b_dpt: applicant['dpt'],
                             u_b_mrk: applicant['mrk'],
                             u_b_job: applicant['job'],
                             u_b_cvr: applicant['cvr']})

def MB_fair_eval(model,applicant,niter):
    return [model.eval(feed_dict={u_b_dpt: applicant['dpt'],
                             u_b_mrk: applicant['mrk'],
                             u_b_cvr: applicant['cvr']}) for _ in range(niter)]
    

# Defining A exogenous nodes
u_a_age = tf.cast(ed.models.Poisson(rate=3.),tf.float32)
u_a_job = tf.cast(ed.models.Bernoulli(probs=.3),tf.float32)
u_a_gnd = tf.cast(ed.models.Bernoulli(probs=.5),tf.float32)
u_a_dpt = tf.cast(ed.models.Categorical(probs=[.7,.2,.1]),tf.float32)
u_a_mrk = ed.models.Beta(concentration0=2.,concentration1=2.)
u_a_cvr = ed.models.Beta(concentration0=2.,concentration1=5.)

# Defining B exogenous nodes
u_b_age = tf.cast(ed.models.Poisson(rate=4.),tf.float32)
u_b_job = tf.cast(ed.models.Bernoulli(probs=.2),tf.float32)
u_b_gnd = tf.cast(ed.models.Bernoulli(probs=.5),tf.float32)
u_b_dpt = tf.cast(ed.models.Categorical(probs=[.7,.15,.1,.05]),tf.float32)
u_b_mrk = ed.models.Beta(concentration0=2.,concentration1=2.)
u_b_cvr = ed.models.Beta(concentration0=2.,concentration1=5.)

# Defining A endogenous nodes
v_a_age = 20. + u_a_age
v_a_gnd = u_a_gnd
v_a_job = u_a_job + tf.truediv(v_a_gnd,.5) + tf.truediv(v_a_age,100.)
v_a_dpt = tf.cond(tf.equal(v_a_gnd,1), lambda: 0., lambda: tf.truediv(u_a_dpt,10.))
v_a_mrk = tf.cond(tf.equal(v_a_dpt,0), lambda: u_a_mrk+.1, lambda:u_a_mrk-.1)
v_a_cvr = u_a_cvr + .2

# Defining A predictor
y_a = v_a_job + v_a_dpt + v_a_mrk + v_a_cvr

# Defining B endogenous nodes
v_b_age = 19. + u_b_age
v_b_gnd = u_b_gnd
v_b_job = u_b_job + tf.truediv(v_b_age,100.) + tf.cond(tf.equal(v_b_gnd,1), lambda:.5, lambda:0.)
v_b_dpt = tf.truediv(u_b_dpt,10.)
v_b_mrk = tf.cond(tf.equal(v_b_dpt,0), lambda: u_b_mrk+.1, lambda:u_b_mrk)
v_b_cvr = u_b_cvr + .1

# Defining B predictor
y_b = tf.truediv(v_b_age,100.) + v_b_job + v_b_dpt + v_b_mrk + v_b_cvr

# Run a session
ed.get_session()

# Sample from Y
y_a_values = [y_a.eval() for _ in range(nsamples)]
y_b_values = [y_b.eval() for _ in range(nsamples)]

# Plot hist and kde of P(Y) 
fig,axes = plt.subplots(2,1)

kde = stats.kde.gaussian_kde(y_a_values)
domain = np.linspace(np.min(y_a_values),np.max(y_a_values), 10000)
axes[0].hist(y_a_values)
axes[0].twinx().plot(domain,kde(domain),'r')
axes[0].axvline(np.mean(y_a_values), color='k', linestyle='dashed', linewidth=1)
axes[0].set_title('P(Y) for Alice')

kde = stats.kde.gaussian_kde(y_b_values)
domain = np.linspace(np.min(y_b_values),np.max(y_b_values), 10000)
axes[1].hist(y_b_values)
axes[1].twinx().plot(domain,kde(domain),'r')
axes[1].axvline(np.mean(y_b_values), color='k', linestyle='dashed', linewidth=1)
axes[1].set_title('P(Y) for Bob')



# Define applicant 1 and 2
app1 = {'age':22, 'gnd': 1., 'dpt': 0., 'mrk':.8, 'job': 1., 'cvr': .4}
app2 = app1.copy(); app2['gnd'] = 0.

# Evaluate applicants in the experts' (unfair) graphs
scoreA_app1 = MA_standard_eval(y_a, app1)
scoreB_app1 = MB_standard_eval(y_b, app1)
scoreA_app2 = MA_standard_eval(y_a, app2)
scoreB_app2 = MB_standard_eval(y_b, app2)

print("Score for applicant 1 from the original causal models: ")
print('SCORE A: {0}'.format(scoreA_app1))
print('SCORE B: {0}'.format(scoreB_app1))
print("Score for applicant 2 from the original causal models: ")
print('SCORE A: {0}'.format(scoreA_app2))
print('SCORE B: {0}'.format(scoreB_app2))


# Evaluate applicant 1 in the experts' (fair) graphs
scoresA_app1 = MA_fair_eval(y_a, app1, nsamples)
scoresB_app1 = MB_fair_eval(y_b, app1, nsamples)
scoresA_app2 = MA_fair_eval(y_a, app2, nsamples)
scoresB_app2 = MB_fair_eval(y_b, app2, nsamples)
print("Score for applicant 1 from the fair causal models: ")
print('SCORE A: {0}'.format(np.average(scoresA_app1)))
print('SCORE B: {0}'.format(np.average(scoresB_app1)))
print("Score for applicant 2 from the fair causal models: ")
print('SCORE A: {0}'.format(np.average(scoresA_app2)))
print('SCORE B: {0}'.format(np.average(scoresB_app2)))


# Plot hist and kde of P(Y|Z=z) 
fig,axes = plt.subplots(2,1)

kde = stats.kde.gaussian_kde(scoresA_app1)
domain = np.linspace(np.min(scoresA_app1),np.max(scoresA_app1), 10000)
axes[0].hist(scoresA_app1)
axes[0].twinx().plot(domain,kde(domain),'r')
axes[0].axvline(np.mean(scoresA_app1), color='k', linestyle='dashed', linewidth=1)
axes[0].set_title('P(Y|Dpt=CS, Mrk=8, Cvr=0.4) for Alice')

kde = stats.kde.gaussian_kde(scoresB_app1)
domain = np.linspace(np.min(scoresB_app1),np.max(scoresB_app1), 10000)
axes[1].hist(scoresB_app1)
axes[1].twinx().plot(domain,kde(domain),'r')
axes[1].axvline(np.mean(scoresB_app1), color='k', linestyle='dashed', linewidth=1)
axes[1].set_title('P(Y|Dpt=CS, Mrk=8, Cvr=0.4) for Bob')

# Plot hist and kde of P(Y|Z=z) 
fig,axes = plt.subplots(2,1)

kde = stats.kde.gaussian_kde(scoresA_app2)
domain = np.linspace(np.min(scoresA_app2),np.max(scoresA_app2), 10000)
axes[0].hist(scoresA_app2)
axes[0].twinx().plot(domain,kde(domain),'r')
axes[0].axvline(np.mean(scoresA_app2), color='k', linestyle='dashed', linewidth=1)
axes[0].set_title('P(Y|Dpt=CS, Mrk=8, Cvr=0.4) for Alice')

kde = stats.kde.gaussian_kde(scoresB_app2)
domain = np.linspace(np.min(scoresB_app2),np.max(scoresB_app2), 10000)
axes[1].hist(scoresB_app2)
axes[1].twinx().plot(domain,kde(domain),'r')
axes[1].axvline(np.mean(scoresB_app2), color='k', linestyle='dashed', linewidth=1)
axes[1].set_title('P(Y|Dpt=CS, Mrk=8, Cvr=0.4) for Bob')


plt.show()