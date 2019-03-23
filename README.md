# Content-Aware Fill for Sequenced Music

<h2>Motivation</h2>
<p>One of the most socially significant effects the computer era has had on the world is its influence on arts and culture. The next frontier lies in the transformation of the computer from simply being the medium of composition to having an active voice in the composition process, turning it into a dialogue. This work is an incremental step toward that goal by exploring how a computer can assist in the composition process by creating a human-in-the-loop music generation tool.</p>

<h2>Goals</h2>
<p>Artists place notes in a piano roll interface when sequencing digital music. After selecting a time range <em>w</em> 8<sup>th</sup> notes wide, the computer should fill in the range with notes. The filled notes should have the following properties:</p>
<ul>
  <li>There should be a smooth transition from the notes preceding the filled region to the filled region and from the filled region to the notes following.</li>
  <li>The notes in the filled region should stylistically match the musical context it inhabits.</li>
</ul>

<img src="https://user-images.githubusercontent.com/5315059/54867745-2e424080-4d5a-11e9-834d-434ec5f10b7c.png" />

<h2>Listen</h2>
Results are from the more successful model, LSTM+WFC.
<table>
<tr><td rowspan="2" width="400"><img src="https://github.com/davepagurek/audio-gap-filling/blob/master/lstm/img/6.png?raw=true" width="400" /></td><td height="180">Ground truth</td></tr>
<tr><td><a href="https://github.com/davepagurek/audio-gap-filling/blob/master/lstm/output/6/output.wav?raw=true">Filled (download .wav)</a></td></tr>
<tr><td rowspan="2" width="400"><img src="https://github.com/davepagurek/audio-gap-filling/blob/master/lstm/img/5.png?raw=true" width="400" /></td><td height="180">Ground truth</td></tr>
<tr><td><a href="https://github.com/davepagurek/audio-gap-filling/blob/master/lstm/output/5/output.wav?raw=true">Filled (download .wav)</a></td></tr>
</table>

<h2>Data</h2>
<p>Models were trained and evaluated on the piano tracks of the Lakh Pianoroll Dataset, which contains 21,425 multitrack songs. Each piano track is a matrix representing whether note <em>i</em> (where <em>i</em> &isin; [0, 127] is a MIDI pitch with semitone increments) is being sounded at time <em>t</em> (where <em>t</em> is in 8<sup>th</sup> notes.)</p>

<img src="https://user-images.githubusercontent.com/5315059/54867748-3dc18980-4d5a-11e9-9531-3d1c8c10d7d5.png" width="60%" />

<h2>Wave Function Collapse</h2>
<p>WFC models divide the input into tiles and create a constraint satisfaction problem when predicting the value of an unknown tile. Two constraints get enforced:</p>

<ol>
  <li>Every predicted tile must only be seen adjacent to another tile if the values were seen in the same configuration somewhere in the known region of the grid.</li>
  <li>The distribution of predicted values should match the distribution of values in the known region of the grid.</li>
</ol>

<p>Each unknown tile, before it is predicted, is thought of as being the superposition of all possible tile options. A tile option is removed if it has never been seen before in its current configuration with adjacent tiles. When a possible tile is selected, the tile is <em>collapsed</em> to that value, and options for surrounding tiles may become impossible.</p>

<p>While there are remaining tile slots to collapse:</p>
<ol>
  <li>Select the tile with lowest Shannon entropy.</li>
  <li>If it has options, randomly collapse the tile to one of its possible options, where each option has a weight proportional to its distribution in the input notes. Propagate the change to adjacent tiles, update their options.</li>
  <li>If it has no options, backtrack a random number of choices and try again.</li>
</ol>

<h2>Long Short-Term Memory with Markov Chain Monte Carlo</h2>

<p>A trained LSTM network generates notes based on a prefix of 200 notes. This is used to fill a target region. Individual LSTM nodes in the network have a height of 128 since each input and output is a vector holding information for every pitch. Each output <em>y</em><sub><em>i</em></sub> represents a vector of probabilities of each pitch being sounded. A boolean vector is sampled for each pitch from these distributions to construct the next time step.</p>

<img src="https://user-images.githubusercontent.com/5315059/54867750-40bc7a00-4d5a-11e9-8131-6396e3708336.png" />

<p>The LSTM has an output space of possible fillings. The filling that maximizes the likelihood of the suffix <em>s</em> to the filled region is found using Markov Chain Monte Carlo sampling:</p>
<ol>
  <li>Initialize <em>f</em><sub>0</sub> by generating a filled region</li>
  <li>For <em>i</em> = 0 to <em>N</em>:
    <ol>
      <li>Sample <em>u</em> ~ U(0, 1).</li>
      <li>Sample <em>n</em> ~ U{1, <em>w</em>}.</li>
      <li>Sample <em>f</em><sub>*</sub> by rewinding <em>f</em><sub><em>i</em>-1</sub> by <em>n</em> 8<sup>th</sup> notes and regenerating <em>n</em> new 8<sup>th</sup> notes.</li>
      <li>If <em>u</em> &lt; (<span>P(<em>s</em>|<em>f</em><sub>*</sub>)P(<em>f</em><sub>*</sub>)</span>) / (<span>P(<em>s</em>|<em>f</em><sub><em>i</em>-1</sub>)P(<em>f</em><sub><em>i</em>-1</sub>)</span>), set <em>f</em><sub><em>i</em></sub> = <em>f</em><sub>*</sub>.</li>
        <li>Otherwise, set <em>f</em><sub><em>i</em></sub> = <em>f</em><sub><em>i</em>-1</sub>.</li>
    </ol>
  </li>
</ol>

<h2>Results</h2>

<p>A piano roll can be thought of as a Markov chain, where the notes being sounded at time <em>t</em> are a state of a Markov process. The transitions between states have probabilities corresponding to how often those two states occur next to each other. States can use pitch number mod 12 to be invariant to the octave of the pitch.</p>

<p>The piano roll from the filled region is modelled as one Markov process and the rest of the piano roll is modelled as another. Assuming piano rolls can be thought of as both being products of the same Markov process, the difference between two processes can be used as a proxy for stylistic fit. Markov process transition probabilities from the filled regions in the ground truth, from WFC, and from LSTM+MCMC were compared against the transition probabilities from the Markov process for the surrounding notes.</p>

<img src="https://user-images.githubusercontent.com/5315059/54867752-474af180-4d5a-11e9-96fb-e9d6aa91403f.png" />

<ul>
  <li><strong>The LSTM+MCMC model produces a similar median error to the ground truth.</strong></li>
  <li>Neither reaches zero due to noise from the filled region not having many samples.</li>
  <li>Both learned models have less variance than the ground truth: real human composers are not perfect Markov processes and can produce "surprising" scores.</li>
  <li>Both have issues continuing obvious structured, repeating patterns.</li>
</ul>
