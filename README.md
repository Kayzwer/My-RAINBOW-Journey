# My-RAINBOW-Journey
<pre>
**Refill Buffer Method**

for i = 0 to n
  fill up buffer
  for j = 0 to m
    sample k data from buffer
    update network parameter
  update target network
  clean up buffer
</pre>

<pre>
**CRBED - Confidence Reward Based Epsilon Decay**

c - confidence
rt - reward target
rt_gr - reward target grow rate
eps - epsilon
eps_dr - epsilon decrease rate
min_eps - minimum epsilon

if last_game_score >= rt:
  confidence_count += 1
  if confidence_count == c:
    eps = (eps - eps_dr) if eps > min_eps else min_eps 
    rt += rt_gr
    confidence_count = 0
else:
  confidence_count = 0
</pre>
