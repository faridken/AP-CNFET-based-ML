simulator lang=spectre


parameters delta_res = 0 
  
statistics {
        process{
             vary delta_res dist=gauss std=10  
           }
        mismatch {
             vary delta_res dist=gauss std=1
           }
 }
simulator lang=spice

.subckt res_ PLUS MINUS
+r=100

.model res_ resistor
R1 (PLUS MINUS) res_ r=r+delta_res
.ends res_


