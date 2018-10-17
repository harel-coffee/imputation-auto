package com.alphame.imputation.client;
import com.google.gwt.core.client.JavaScriptObject;

public class AvailableData extends JavaScriptObject {                              // (1)
  // Overlay types always have protected, zero argument constructors.
  protected AvailableData() {}                                              // (2)

  // JSNI methods to get stock data.
  public final native String get_landscape_name() /*-{ return this.landscape_name; }-*/; // (3)
  }


