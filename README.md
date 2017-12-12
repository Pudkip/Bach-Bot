<div tabindex="-1" id="notebook" class="border-box-sizing">

<div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

# Baroque Music Generator[¶](#Bach-Music-Generator)

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Overview[¶](#Overview)

This aim of this project is to use a time series neural network to generate music trained on 15 Chorales by Bach. A lot of this involves parsing information from midi files into the notes for the melody. There is not a lot of info for processing midi files...at all. So to start off, I'm only predicting the notes from the melody (as opposed to bass, rythm, or whatever the different layers are called) and without time i.e. every note is assumed to last for the same time.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Processing Midi Files[¶](#Processing-Midi-Files)

Some functions I wrote for parsing the notes from the melody in the track into something the network can process and then back into a midi message.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [1]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">mido</span>

<span class="k">def</span> <span class="nf">decodeMidi</span><span class="p">(</span><span class="n">midifile</span><span class="p">,</span><span class="n">num_layers</span><span class="o">=</span><span class="mi">2</span><span class="p">):</span>
    <span class="n">song</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">track</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">midifile</span><span class="o">.</span><span class="n">tracks</span><span class="p">):</span>
        <span class="n">song</span><span class="o">.</span><span class="n">append</span><span class="p">([])</span>
        <span class="k">for</span> <span class="n">msg</span> <span class="ow">in</span> <span class="n">track</span><span class="p">:</span>
            <span class="n">message</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
            <span class="k">if</span> <span class="s1">'<meta'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span> <span class="ow">and</span> <span class="s1">'control_change'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span> <span class="ow">and</span> <span class="s1">'program_change'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span><span class="p">:</span>
                <span class="n">channel</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"="</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
                <span class="n">note</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"="</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
                <span class="n">velocity</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()[</span><span class="mi">3</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"="</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
                <span class="n">time</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()[</span><span class="mi">4</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">'='</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
                <span class="n">song</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">append</span><span class="p">([</span><span class="n">channel</span><span class="p">,</span> <span class="n">note</span><span class="p">,</span> <span class="n">velocity</span><span class="p">,</span> <span class="n">time</span><span class="p">])</span>

    <span class="n">song</span> <span class="o">=</span> <span class="p">[</span><span class="n">x</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">song</span> <span class="k">if</span> <span class="n">x</span><span class="p">]</span>
    <span class="n">song</span> <span class="o">=</span> <span class="p">[</span><span class="n">song</span><span class="p">[:</span><span class="n">num_layers</span><span class="p">]]</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">song</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">encodeMidi</span><span class="p">(</span><span class="n">song</span><span class="p">):</span>
    <span class="n">file</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiFile</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">])):</span>
        <span class="n">track</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiTrack</span><span class="p">()</span>
        <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'control_change'</span><span class="p">,</span> <span class="n">channel</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">control</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">value</span><span class="o">=</span><span class="mi">80</span><span class="p">,</span> <span class="n">time</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
        <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'control_change'</span><span class="p">,</span> <span class="n">channel</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">control</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span> <span class="n">value</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">time</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
        <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'program_change'</span><span class="p">,</span> <span class="n">channel</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">program</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">time</span><span class="o">=</span><span class="mi">0</span><span class="p">))</span>
        <span class="k">for</span> <span class="n">j</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">])):</span>
            <span class="n">note</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'note_on'</span><span class="p">,</span><span class="n">channel</span><span class="o">=</span><span class="nb">int</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">j</span><span class="p">][</span><span class="mi">0</span><span class="p">]),</span> <span class="n">note</span><span class="o">=</span><span class="nb">int</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">j</span><span class="p">][</span><span class="mi">1</span><span class="p">]),</span> <span class="n">velocity</span><span class="o">=</span><span class="nb">int</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">j</span><span class="p">][</span><span class="mi">2</span><span class="p">]),</span> <span class="n">time</span><span class="o">=</span><span class="nb">int</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="n">i</span><span class="p">][</span><span class="n">j</span><span class="p">][</span><span class="mi">3</span><span class="p">]))</span>
            <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">note</span><span class="p">)</span>
        <span class="n">file</span><span class="o">.</span><span class="n">tracks</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">track</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">file</span>

<span class="k">def</span> <span class="nf">midiToNote</span><span class="p">(</span><span class="n">midifile</span><span class="p">,</span><span class="n">num_layers</span><span class="o">=</span><span class="mi">2</span><span class="p">):</span>
    <span class="n">song</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">track</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">midifile</span><span class="o">.</span><span class="n">tracks</span><span class="p">):</span>
        <span class="n">song</span><span class="o">.</span><span class="n">append</span><span class="p">([])</span>
        <span class="k">for</span> <span class="n">msg</span> <span class="ow">in</span> <span class="n">track</span><span class="p">:</span>
            <span class="n">message</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
            <span class="k">if</span> <span class="s1">'<meta'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span> <span class="ow">and</span> <span class="s1">'control_change'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span> <span class="ow">and</span> <span class="s1">'program_change'</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">message</span><span class="p">:</span>
                <span class="n">note</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()[</span><span class="mi">2</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"="</span><span class="p">)[</span><span class="o">-</span><span class="mi">1</span><span class="p">])</span>
                <span class="n">song</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">.</span><span class="n">append</span><span class="p">([</span><span class="n">note</span><span class="p">])</span>

    <span class="n">song</span> <span class="o">=</span> <span class="p">[</span><span class="n">x</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">song</span> <span class="k">if</span> <span class="n">x</span><span class="p">]</span>
    <span class="n">song</span> <span class="o">=</span> <span class="p">[</span><span class="n">song</span><span class="p">[:</span><span class="n">num_layers</span><span class="p">]]</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">song</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">predictionsToNotes</span><span class="p">(</span><span class="n">preds</span><span class="p">):</span>
    <span class="n">song</span> <span class="o">=</span> <span class="n">preds</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">song</span> <span class="o">=</span> <span class="n">song</span><span class="o">.</span><span class="n">tolist</span><span class="p">()</span>
    <span class="n">_</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">noteIndex</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">song</span><span class="p">:</span>
        <span class="n">best</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">song</span><span class="p">[</span><span class="n">_</span><span class="p">])</span>
        <span class="n">key</span> <span class="o">=</span> <span class="n">song</span><span class="p">[</span><span class="n">_</span><span class="p">]</span><span class="o">.</span><span class="n">index</span><span class="p">(</span><span class="n">best</span><span class="p">)</span>
        <span class="n">noteIndex</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">key</span><span class="p">)</span>
        <span class="c1"># print(best,key)</span>
        <span class="n">_</span> <span class="o">+=</span> <span class="mi">1</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">noteIndex</span><span class="p">)</span>

<span class="k">def</span> <span class="nf">notesToMidi</span><span class="p">(</span><span class="n">notes</span><span class="p">,</span> <span class="n">velocity</span> <span class="o">=</span> <span class="mi">95</span><span class="p">,</span> <span class="n">time</span> <span class="o">=</span> <span class="mi">116</span><span class="p">):</span>
    <span class="n">file</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiFile</span><span class="p">()</span>
    <span class="n">track</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiTrack</span><span class="p">()</span>
    <span class="n">file</span><span class="o">.</span><span class="n">tracks</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">track</span><span class="p">)</span>
    <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'program_change'</span><span class="p">,</span> <span class="n">program</span><span class="o">=</span><span class="mi">12</span><span class="p">,</span> <span class="n">time</span><span class="o">=</span><span class="n">time</span><span class="p">))</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">notes</span><span class="p">)):</span>
        <span class="n">track</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">mido</span><span class="o">.</span><span class="n">Message</span><span class="p">(</span><span class="s1">'note_on'</span><span class="p">,</span> <span class="n">note</span><span class="o">=</span><span class="n">notes</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">velocity</span><span class="o">=</span><span class="n">velocity</span><span class="p">,</span> <span class="n">time</span><span class="o">=</span><span class="n">time</span><span class="p">))</span>
    <span class="k">return</span><span class="p">(</span><span class="n">file</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Here I process the files, padding each track to 1000 messages (notes), labels are the subsequent notes.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [2]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">from</span> <span class="nn">processingData</span> <span class="k">import</span> <span class="n">decodeMidi</span>
<span class="kn">import</span> <span class="nn">mido</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">os</span>

<span class="n">files</span> <span class="o">=</span> <span class="n">os</span><span class="o">.</span><span class="n">listdir</span><span class="p">(</span><span class="s1">'train'</span><span class="p">)</span>
<span class="n">features</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1000</span><span class="p">,</span><span class="mi">88</span><span class="p">))</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1000</span><span class="p">,</span><span class="mi">88</span><span class="p">))</span>
<span class="k">for</span> <span class="n">file</span> <span class="ow">in</span> <span class="n">files</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">file</span><span class="p">)</span>
    <span class="n">mid</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiFile</span><span class="p">(</span><span class="s2">"train/"</span><span class="o">+</span><span class="n">file</span><span class="p">)</span>
    <span class="n">mid</span> <span class="o">=</span> <span class="n">decodeMidi</span><span class="p">(</span><span class="n">mid</span><span class="p">)</span>
    <span class="n">mid</span> <span class="o">=</span> <span class="n">mid</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">featuresPart</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">labelsPart</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1001</span><span class="p">):</span>
        <span class="k">if</span> <span class="n">i</span> <span class="o"><</span> <span class="nb">len</span><span class="p">(</span><span class="n">mid</span><span class="p">):</span>
            <span class="n">onehot</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">88</span>
            <span class="n">onehot</span><span class="p">[</span><span class="n">mid</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="mi">1</span><span class="p">]]</span> <span class="o">=</span> <span class="mi">1</span>
            <span class="n">featuresPart</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">onehot</span><span class="p">)</span>
            <span class="n">onehot</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">88</span>
            <span class="k">try</span><span class="p">:</span>
                <span class="n">onehot</span><span class="p">[</span><span class="n">mid</span><span class="p">[</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">][</span><span class="mi">1</span><span class="p">]]</span> <span class="o">=</span> <span class="mi">1</span>
                <span class="n">labelsPart</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">onehot</span><span class="p">)</span>
            <span class="k">except</span> <span class="ne">Exception</span><span class="p">:</span>
                <span class="k">pass</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">pad</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">88</span>
            <span class="n">featuresPart</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">pad</span><span class="p">)</span>
            <span class="n">labelsPart</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">pad</span><span class="p">)</span>
    <span class="n">featuresPart</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">featuresPart</span><span class="p">[:</span><span class="mi">1000</span><span class="p">]])</span>
    <span class="n">labelsPart</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">labelsPart</span><span class="p">[:</span><span class="mi">1000</span><span class="p">]])</span>
    <span class="n">features</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">((</span><span class="n">features</span><span class="p">,</span><span class="n">featuresPart</span><span class="p">))</span>
    <span class="n">labels</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">concatenate</span><span class="p">((</span><span class="n">labels</span><span class="p">,</span><span class="n">labelsPart</span><span class="p">))</span>

<span class="n">features</span> <span class="o">=</span> <span class="n">features</span><span class="p">[</span><span class="mi">1</span><span class="p">:]</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">labels</span><span class="p">[</span><span class="mi">1</span><span class="p">:]</span>
<span class="nb">print</span><span class="p">(</span><span class="n">features</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">labels</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>01AusmeinesHerz.mid
02Ichdankdir.mid
03AchGott.mid
04EsistdasHeiluns.mid
05AnWasserflussen.mid
06Christus.mid
07Nunlob.mid
08Freueteuch.mid
09Ermuntredich.mid
10AustieferNot.mid
11Jesu.mid
12PuerNatusinBet.mid
13Alleinzudir.mid
14OHerreGott.mid
15ChristlaginTode.mid
(15, 1000, 88)
(15, 1000, 88)
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Building and Training the Model[¶](#Building-and-Training-the-Model)

Use Keras' simple RNN to train the model.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [3]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="o">%%</span><span class="k">capture</span>
from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense

from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(SimpleRNN(input_dim  =  88, output_dim = 88, return_sequences = True))
model.add(TimeDistributed(Dense(output_dim = 88, activation  =  "softmax")))
model.compile(loss = "mse", optimizer = "rmsprop", metrics=['accuracy'])
model.fit(features, labels,
          epochs=1000,
          batch_size=256,
          callbacks=[ModelCheckpoint("Simple_RNN_3", monitor='val_acc',save_best_only=True)])
model.save("Simple_RNN_3.h5")
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Generating Music from Beethoven's Moonlight Sonata[¶](#Generating-Music-from-Beethoven's-Moonlight-Sonata)

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [4]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">from</span> <span class="nn">keras.models</span> <span class="k">import</span> <span class="n">load_model</span>
<span class="kn">from</span> <span class="nn">processingData</span> <span class="k">import</span> <span class="n">decodeMidi</span><span class="p">,</span> <span class="n">encodeMidi</span>
<span class="kn">import</span> <span class="nn">mido</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pickle</span>

<span class="n">song</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">MidiFile</span><span class="p">(</span><span class="s2">"moonlightSonata.mid"</span><span class="p">)</span>
<span class="n">song</span> <span class="o">=</span> <span class="n">decodeMidi</span><span class="p">(</span><span class="n">song</span><span class="p">)</span>
<span class="n">song</span> <span class="o">=</span> <span class="n">song</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
<span class="n">test</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">song</span><span class="p">)):</span>
    <span class="n">onehot</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">88</span>
    <span class="n">onehot</span><span class="p">[</span><span class="n">song</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="mi">1</span><span class="p">]]</span> <span class="o">=</span> <span class="mi">1</span>
    <span class="n">test</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">onehot</span><span class="p">)</span>
    <span class="n">onehot</span> <span class="o">=</span> <span class="p">[</span><span class="mi">0</span><span class="p">]</span> <span class="o">*</span> <span class="mi">88</span>

<span class="n">test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">test</span><span class="p">[:</span><span class="mi">1000</span><span class="p">]])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">test</span><span class="o">.</span><span class="n">shape</span><span class="p">)</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">load_model</span><span class="p">(</span><span class="s1">'Simple_RNN_3.h5'</span><span class="p">)</span>
<span class="n">prediction</span> <span class="o">=</span> <span class="p">[</span><span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test</span><span class="p">)]</span>

<span class="n">savePredictions</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="s2">"moonlightPrediction.pickle"</span><span class="p">,</span><span class="s2">"wb"</span><span class="p">)</span>
<span class="n">pickle</span><span class="o">.</span><span class="n">dump</span><span class="p">(</span><span class="n">prediction</span><span class="p">,</span> <span class="n">savePredictions</span><span class="p">)</span>
<span class="n">savePredictions</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>(1, 236, 88)
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Converting Results to Midi[¶](#Converting-Results-to-Midi)

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [5]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">pickle</span>
<span class="kn">from</span> <span class="nn">processingData</span> <span class="k">import</span> <span class="n">predictionsToNotes</span><span class="p">,</span> <span class="n">notesToMidi</span>
<span class="kn">import</span> <span class="nn">mido</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="n">pred</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="nb">open</span><span class="p">(</span><span class="s1">'moonlightPrediction.pickle'</span><span class="p">,</span><span class="s1">'rb'</span><span class="p">))</span>
<span class="n">pred</span> <span class="o">=</span> <span class="n">predictionsToNotes</span><span class="p">(</span><span class="n">pred</span><span class="p">)</span>
<span class="n">file</span> <span class="o">=</span> <span class="n">notesToMidi</span><span class="p">(</span><span class="n">pred</span><span class="p">,</span><span class="n">time</span><span class="o">=</span><span class="mi">60</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">file</span><span class="p">)</span>

<span class="n">file</span><span class="o">.</span><span class="n">ticks_per_beat</span> <span class="o">=</span> <span class="mi">120</span>
<span class="n">port</span> <span class="o">=</span> <span class="n">mido</span><span class="o">.</span><span class="n">open_output</span><span class="p">()</span>

<span class="k">for</span> <span class="n">msg</span> <span class="ow">in</span> <span class="n">file</span><span class="o">.</span><span class="n">play</span><span class="p">():</span>
    <span class="n">port</span><span class="o">.</span><span class="n">send</span><span class="p">(</span><span class="n">msg</span><span class="p">)</span>

</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre><midi file None type 1, 1 tracks, 237 messages>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Here's what it sounds like:[¶](#Here's-what-it-sounds-like:)

</div>

<audio controls=""><source src="https://github.com/Pudkip/Bach-Bot/blob/master/moonlight_sonata_bach.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

...and compared to the stripped, time-constant version of the original:

<audio controls=""><source src="https://github.com/Pudkip/Bach-Bot/blob/master/moonlight_sonata_augmented.mp3" type="audio/mpeg"> Your browser does not support the audio element.</audio>

</div>

</div>

</div>

</div>
