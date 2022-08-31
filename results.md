---
layout: page
title: <a href="https://rocknroll87q.github.io/LOD-Brain/">LOD-Brain</a>
---

[<-- main page](https://rocknroll87q.github.io/LOD-Brain/)

<hr>

## First result


<script src="https://unpkg.com/@niivue/niivue@0.29.0/dist/niivue.umd.js"></script>

<canvas id="gl" height=600></canvas>

<script>
  var volumeList = [
    // first object in array is background image
      {
        url: "./results/MALC2012_1000_3_256iso_t1.nii.gz",
        url: "./results/MALC2012_1000_3_256iso_predicted_volume.nii.gz",
      }
   ]

 // Niivue will adjust the canvas to 100% of its parent container's size 
 // the parent element can be any size you want (small or large)
 var nv = new niivue.Niivue()
 nv.attachTo('gl') // the canvas element id
 nv.loadVolumes(volumeList)
 nv.setSliceType(nv.sliceTypeMultiPlanar) // press the "v" key to cycle through views
</script>









