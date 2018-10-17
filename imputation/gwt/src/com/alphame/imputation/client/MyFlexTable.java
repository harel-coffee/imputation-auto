package com.alphame.imputation.client;
import com.google.gwt.event.dom.client.HasMouseMoveHandlers;
import com.google.gwt.event.dom.client.HasMouseOutHandlers;
import com.google.gwt.event.dom.client.HasMouseOverHandlers;
import com.google.gwt.event.dom.client.MouseMoveEvent;
import com.google.gwt.event.dom.client.MouseMoveHandler;
import com.google.gwt.event.dom.client.MouseOutEvent;
import com.google.gwt.event.dom.client.MouseOutHandler;
import com.google.gwt.event.dom.client.MouseOverEvent;
import com.google.gwt.event.dom.client.MouseOverHandler;
import com.google.gwt.event.shared.HandlerRegistration;
import com.google.gwt.user.client.DOM;
import com.google.gwt.dom.client.Element;
import com.google.gwt.user.client.Event;
import com.google.gwt.user.client.ui.FlexTable;


public  class MyFlexTable extends FlexTable implements HasMouseOverHandlers, HasMouseOutHandlers ,HasMouseMoveHandlers{
    /*
	private Element head;  
    private Element headerTr;
    */  
	public MyFlexTable(){
		super();
		/*
	    head = DOM.createTHead();  
	    headerTr = DOM.createTR();  
	    DOM.insertChild(this.getElement(), head, 0);  
	    DOM.insertChild(head, headerTr, 0);  
	    Element tBody = getBodyElement();  
	    DOM.setElementAttribute(tBody, "style", "overflow:auto;text-align: left;");  
	    DOM.setElementAttribute(head, "style", "text-align: left;background-color: #2062B8;");
	    */  
		sinkEvents(Event.ONMOUSEDOWN |Event.ONMOUSEUP |Event.ONMOUSEOVER |Event.ONMOUSEOUT);       
    }
	
	@Override
	public HandlerRegistration addMouseOutHandler(MouseOutHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseOutEvent.getType());
	}
	
	@Override
	public HandlerRegistration addMouseMoveHandler(MouseMoveHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseMoveEvent.getType());
	}
	
	@Override
	public HandlerRegistration addMouseOverHandler(MouseOverHandler handler) {
		// TODO Auto-generated method stub
		return addDomHandler(handler, MouseOverEvent.getType());
	}
	
	public Element getRowForMouseMoveEvent(MouseMoveEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	public Element getRowForMouseOverEvent(MouseOverEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	public Element getRowForMouseOutEvent(MouseOutEvent event) {
	    Element td = getEventTargetCell(Event.as(event.getNativeEvent()));
	    if (td == null) {
	      return null;
	    }
	    Element tr = DOM.getParent(td);
	    if (tr == null){
		      return null;
	    }	    	
	    return tr;
	}
	
	
    @Override
	public void onBrowserEvent(Event event) {
        Element td = getEventTargetCell(event);
        if (td == null) return;
        Element tr = DOM.getParent(td);
        switch (DOM.eventGetType(event)) {
        		
        		case Event.ONMOUSEDOWN: {
        				tr.getStyle().setProperty("backgroundColor", "#F88017");      
                        break;
                }
        		/*
                case Event.ONMOUSEUP: {
                        DOM.setStyleAttribute(tr, "backgroundColor", "#ffffff");
                        break;
                }
                */                
                case Event.ONMOUSEOVER: {
                		//tr.getStyle().setProperty("backgroundColor", "lightslategray");
                		//tr.getStyle().setProperty("cursor", "pointer");
                        //break;
                }
                case Event.ONMOUSEOUT: {
                		tr.getStyle().setProperty("backgroundColor","#ADDFFF");
                        break;
                }              
       }
       super.onBrowserEvent(event);
    }       
}
   
     
     /* 
    public void setHeight(String height) {  
     DOM.setElementAttribute(getBodyElement(), "height", height);  
    }  
      
    public void setHeader(int column,String text){  
     prepareHeader(column);  
        if (text != null) {  
           DOM.setInnerText(DOM.getChild(headerTr, column), text);  
     }  
    }  
      
    private void prepareHeader(int column) {  
     if (column < 0) {  
            throw new IndexOutOfBoundsException(  
                "Cannot create a column with a negative index: " + column);  
      }  
      int cellCount = DOM.getChildCount(headerTr);  
         int required = column + 1 - cellCount;  
         if (required > 0) {  
           addCells(head, 0, required);  
         }  
    }  
      
      
    public void setHeaderWidget(int column, Widget widget) {  
       prepareHeader(column);  
        if (widget != null) {  
          widget.removeFromParent();  
          // Physical attach.  
          DOM.appendChild(DOM.getChild(headerTr, column), widget.getElement());  
      
          adopt(widget);  
        }  
      }        
    private native void addCells(Element table, int row, int num)/*-{ 
	    var rowElem = table.rows[row]; 
	    for(var i = 0; i < num; i++){ 
	      var cell = $doc.createElement("td"); 
	      rowElem.appendChild(cell);   
	    } 
 	};
 	*/

 