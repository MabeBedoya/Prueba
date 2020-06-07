import pandas as pd
import plotly
import plotly.plotly 
from plotly.graph_objs import *
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import scipy
import networkx

			
#====================== Leer Archivos ==============================#
  
demandDF = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/DemandDataOct.xlsx')


pasillos = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/ubicacionPasillo.xlsx','pasillos')


palletDF1 = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/Pallets112-441.xlsx')



#======================= ESTO YA ES DEL PALLET COMO TAL =============================

#============== LOCALES =======================
Locales= pd.DataFrame()
Locales['LOCAL'] = demandDF["LOCAL"]
Locales=Locales.drop_duplicates(subset="LOCAL")
Locales=Locales.sort_values(["LOCAL"])
#==============================================

colors = ['green','blue','red','magenta','orange','cyan','olive','yellow','purple','violet','pink']

distandf= []



for local in Locales['LOCAL']:  
    palletDF=palletDF1[(palletDF1.LOCAL == local)]

    for nave in range(1,6):
        draw = []   #guarda trazos
        colorRuta = 0
        df2 = palletDF[(palletDF.NAVE == nave)] #filtro para tomar info solo de la nave a graficar
        df3 = df2['PALLET']     # filtrar palets de la nave, PALLET-TAREA   
        pallets = df3.drop_duplicates()
        pallets=pallets.to_frame()
        pallets.set_index(scipy.arange(len(pallets)),inplace=True)
        
        dln = []
        
        for i in pallets['PALLET']: # recorre pallets(de la nave)
            dftemp = df2[(df2.PALLET==i)]
            dftemp.set_index(scipy.arange(len(dftemp)),inplace=True)
            
            if len(dftemp)>1: # unicamente los q tienen mas de 1 codigo de barra
                demandTemp = demandDF[demandDF['CODIGO BARRAS'].isin(dftemp.cb.values)] #.sample(40)
                demandTemp['adjPasillo'] = demandTemp.PASILLO.map(lambda pasillo: int(pasillo - 100*int(pasillo/100)))# numero de pasillo sin tener en cta la nave
                demandTemp['adjRack'] = scipy.int64(demandTemp.RACK/10) #no es 390 deja 39
                
                
                
                finesPasillo = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/ubicacionPasillo.xlsx','fines de pasillo') #lee pestaña fines de pasillo 
                pasillos = pasillos[pasillos.nave == nave]
                finesPasillo = finesPasillo[finesPasillo.nave == nave]
                #pdb.set_trace()
                entrada = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/ubicacionPasillo.xlsx','ENTRADA')
                entrada.columns = ['entradaX','entradaY','ubicacion']
                toEntrada = finesPasillo.merge(entrada,how='inner',left_on = 'ubicacion', right_on = 'ubicacion')
                #create an undirected graph
                g = networkx.Graph()
                #add edges from the bottom of every row to the start point
                
                # add_edge-agrega bordes desde la parte inferior de cada fila hasta el punto de inicio
                for row in toEntrada.itertuples(): #conectar entradas con fines de pasillo
                    g.add_edge('finPasillo{}_{}'.format(row.pasillo, row.ubicacion) , 'entrada', distance = scipy.sqrt((row.entradaX - row.x)**2+(row.entradaY - row.y)**2))
                
                # agregue bordes de todos los extremos del pasillo entre sí, ya que tienen la misma ubicación o el mismo pasillo
                meetUbicacion = finesPasillo.merge(finesPasillo,how='inner',left_on='ubicacion',right_on='ubicacion')
                meetUbicacion = meetUbicacion[meetUbicacion.pasillo_x < meetUbicacion.pasillo_y]
                
                for row in meetUbicacion.itertuples():#unir fines de pasillo entre si
                    g.add_edge('finPasillo{}_{}'.format(row.pasillo_x,row.ubicacion), 'finPasillo{}_{}'.format(row.pasillo_y,row.ubicacion), distance = scipy.sqrt((row.x_x - row.x_y)**2+(row.y_x - row.y_y)**2))
                meetPasillo = finesPasillo.merge(finesPasillo,how='inner',left_on='pasillo',right_on='pasillo')
                meetPasillo = meetPasillo[meetPasillo.ubicacion_x < meetPasillo.ubicacion_y]
                
                for row in meetPasillo.itertuples():
                    g.add_edge('finPasillo{}_{}'.format(row.pasillo,row.ubicacion_x), 'finPasillo{}_{}'.format(row.pasillo,row.ubicacion_y), distance = scipy.sqrt((row.x_x - row.x_y)**2+(row.y_x - row.y_y)**2))
                #f = open('prueba.dat', 'w')
                #connect the products to the pasillo edges
                cbPasillo = demandTemp.merge(finesPasillo,how='inner',left_on='adjPasillo',right_on='pasillo')
                #cbPasillo = cbPasillo[['CODIGO BARRAS','x','X','y','Y','adjPasillo','ubicacion']].drop_duplicates()
               
                for idx,row in cbPasillo.iterrows():#conecta pasillo con codigo de barra simepre y cuando esten en la misma linea
                    g.add_edge('finPasillo{}_{}'.format(row.adjPasillo,row.ubicacion), 'cb_{}'.format(row['CODIGO BARRAS']), distance = scipy.sqrt((row.x - row.X)**2+(row.y - row.Y)**2))
                	#writeln('finPasillo{}_{}'.format(row.adjPasillo,row.ubicacion))
                	#writeln('1' + ' cb_{}'.format(row['CODIGO BARRAS']))
                #connect products in the same pasillo to each other
                #unir codigos de barra de un solo pasillo
                cb2cb = demandTemp.merge(demandTemp,how='inner',left_on = 'adjPasillo', right_on = 'adjPasillo')
                cb2cb = cb2cb[cb2cb['CODIGO BARRAS_x'] < cb2cb['CODIGO BARRAS_y']]
                
                for idx,row in cb2cb.iterrows():
                    g.add_edge('cb_{}'.format(row['CODIGO BARRAS_x']), 'cb_{}'.format(row['CODIGO BARRAS_y']), distance = scipy.sqrt((row.X_x - row.X_y)**2+(row.Y_x - row.Y_y)**2))
                
                pathDict={}
                pathDict = dict(networkx.all_pairs_dijkstra_path(g,weight='distance'))
                distances={} #mtx de distancias, calcula la distancias entre los puntos 
                distances = dict(networkx.all_pairs_dijkstra_path_length(g,weight='distance'))
                
                for i in range(len(dftemp['cb'])):
                    dftemp['cb'][i] = 'cb_'+str(dftemp['cb'][i])
                
                coordP = {} #diccionario con las coordenadas 
                for i in range(len(dftemp['cb'])):
                    coordP[dftemp['cb'][i]] = (dftemp['X'][i],dftemp['Y'][i])
                
                graph = networkx.Graph()
                for i in range(len(dftemp['cb'])-1):#conecte un tras otro
                    graph.add_edge(dftemp['cb'][i],dftemp['cb'][i+1],weight = distances[dftemp['cb'][i]][dftemp['cb'][i+1]])
                
                net=networkx.single_source_dijkstra_path_length(graph,dftemp['cb'][0])
                distanciaNet = net[dftemp['cb'][len(dftemp['cb'])-1]]#distancia total de recorrido,orden de codigos de barra
                
                dln.append(distanciaNet)
                
                completa = [] # define la ruta, "cuando llega a fin de pasillo"
                for i in range(len(dftemp['cb'])-1):
                    ruta = pathDict[dftemp['cb'][i]][dftemp['cb'][i+1]]
                    for j in range(len(ruta)-1):
                        completa.append(ruta[j])
                completa.append(dftemp['cb'][len(dftemp['cb'])-1]) #ruta completa, recorrido de todos lo spuntos del pedido completo
                
                coordF = {}
                for i in finesPasillo['pasillo'].index:
                    coordF['finPasillo'+str(finesPasillo['pasillo'][i])+'_'+str(finesPasillo['ubicacion'][i])] = (finesPasillo['x'][i],finesPasillo['y'][i])
                coordenadas = coordP.update(coordF) #coor de fines de pasillos para que pinte
                
                cb = dftemp['cb']
                xp = list(dftemp['X'])
                yp = list(dftemp['Y'])
                index = list(range(1,len(yp)+1))
                
                node_trace=go.Scatter(  #numero q pinta es el orden en el q se recorre
                        x=xp,
                        y=yp,
                        text=index,
                        textfont=dict(color='red',size=10),
                        textposition='bottom center',
                        mode='markers+text',
                        name='orden SmartBP',
                        hoverinfo='text',
                        marker=go.scatter.Marker(
                                color = colors[colorRuta],
                                size = 2
                                )
                        )
                
                draw.append(node_trace)
                
                edge_trace = go.Scatter(
                        x=[],
                        y=[],
                        mode='lines',
                        name='ruta SmartBP dist: '+str(round(distanciaNet,0)),
                        line=go.scatter.Line(
                                color = colors[colorRuta],
                                shape = 'spline',
                                width = 3
                                )
                        )
                
                for i in range(len(completa)-1):
                    x0 = coordP[completa[i]][0]
                    y0 = coordP[completa[i]][1]
                    x1 = coordP[completa[i+1]][0]
                    y1 = coordP[completa[i+1]][1]
                    edge_trace['x'] += (x0, x1)
                    edge_trace['y'] += (y0, y1)
                    edge_trace['line']['width'] = 2
            
                draw.append(edge_trace)
                
                colorRuta += 1 #cambie de color
        
        #extraer pasillos físicos       
        edge_traceEs = go.Scatter(
                x=[],
                y=[],
                mode='lines',
                line=go.scatter.Line(
                        color = 'black',
                        shape = 'spline',
                        width = 3
                        )
                )
        
        df2 = pd.read_excel('C:/Users/mbbedoya/MaBe/SMART BP/Archivos/Pasillos.xlsx')
        df2.set_index(scipy.arange(len(df2)),inplace=True)
            
        xe = df2['X']
        ye = df2['Y']
        estantes = []
        for i in range(len(xe)):
            estantes.append((xe[i],ye[i]))
            
        for i in range(len(estantes)):
            if (i+1)%2 >0:  
                x0 = estantes[i][0]
                y0 = estantes[i][1]
                x1 = estantes[i+1][0]
                y1 = estantes[i+1][1]
                edge_traceEs['x'] += (x0, x1, None)
                edge_traceEs['y'] +=(y0, y1, None)
                edge_traceEs['line']['width'] = 3
        
        draw.append(edge_traceEs)
        
        #Figure(data=None, layout=None, frames=None, skip_invalid=False)    
        fig=go.Figure(data=draw,
                   layout=go.Layout(
                    title='<br>Picking Favorita Local '+str(local)+' Nave ' + str(nave),
                    titlefont=dict(family='Timer New Roman', size=20),
                    xaxis=go.layout.XAxis(showgrid = False, zeroline=False, showticklabels=False),
                    yaxis=go.layout.YAxis(showgrid = False, zeroline=False, showticklabels=False)
                    ))
        
        largo = len(dln)
        LNTD1 = pd.DataFrame()
        LNTD1= pd.DataFrame(columns=('LOCAL', 'NAVE'))
        LNTD1.loc[len(LNTD1)]=[''+str(local),'' + str(nave)]
        LNTD1=pd.concat([LNTD1]*largo)
        LNTD1['TAREA']= np.arange(1,largo+1)
        LNTD1['DISTANCIA']= dln
        distandf.append(LNTD1)
        
        py.iplot(fig, filename='(Nuevo) Rutas de Picking Favorita Local '+str(local)+' Nave ' + str(nave))
    
df= pd.DataFrame([distandf])
df.to_csv('Res112-441.csv', index=False,header=False)    



