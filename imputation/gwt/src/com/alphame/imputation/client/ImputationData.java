package com.alphame.imputation.client;
import com.google.gwt.core.client.JavaScriptObject;

public class ImputationData extends JavaScriptObject {                              // (1)
  // Overlay types always have protected, zero argument constructors.
  protected ImputationData() {}                                              // (2)

  // JSNI methods to get stock data.
  public final native String get_aa_ref() /*-{ return this.aa_ref; }-*/; // (3)
  public final native int get_aa_pos() /*-{ return this.aa_pos; }-*/; // (3)
  public final native int get_aa_pos_index() /*-{ return this.aa_pos_index; }-*/; // (3)
  public final native String get_aa_alt() /*-{ return this.aa_alt; }-*/; // (3)
  
  public final native Double get_fitness_org() /*-{ return this.fitness_org; }-*/; // (3)
  public final native String get_fitness_org_colorcode() /*-{ return this.fitness_org_colorcode; }-*/;
  public final native Double get_fitness_se_org() /*-{ return this.fitness_se_org; }-*/;

  public final native Double get_fitness_reverse() /*-{ return this.fitness_reverse; }-*/; // (3)
  public final native String get_fitness_reverse_colorcode() /*-{ return this.fitness_reverse_colorcode; }-*/;
  public final native Double get_fitness_se_reverse() /*-{ return this.fitness_se_reverse; }-*/;
  
  public final native Double get_fitness_refine() /*-{ return this.fitness_refine; }-*/; // (3)
  public final native String get_fitness_refine_colorcode() /*-{ return this.fitness_refine_colorcode; }-*/;
  public final native Double get_fitness_se_refine() /*-{ return this.fitness_se_refine; }-*/;
  
  public final native Double get_quality_score() /*-{ return this.quality_score; }-*/;
   
  public final native int get_se_refine_fontsize() /*-{ return this.se_refine_fontsize; }-*/;
  public final native int get_se_org_fontsize() /*-{ return this.se_org_fontsize; }-*/;
  
  public final native int get_pseudo_count() /*-{ return this.pseudo_count; }-*/;  
  public final native int get_num_replicates() /*-{ return this.num_replicates; }-*/;

  public final native Double get_polyphen_score() /*-{ return this.polyphen_score; }-*/;
  public final native String get_polyphen_colorcode() /*-{ return this.polyphen_colorcode; }-*/;
  public final native Double get_sift_score() /*-{ return this.sift_score; }-*/;
  public final native String get_sift_colorcode() /*-{ return this.sift_colorcode; }-*/;
  public final native Double get_provean_score() /*-{ return this.provean_score; }-*/;
  public final native String get_provean_colorcode() /*-{ return this.provean_colorcode; }-*/;
  public final native Double get_funsum_fitness_mean() /*-{ return this.funsum_fitness_mean; }-*/;
  public final native Double get_blosum62() /*-{ return this.blosum62; }-*/;
  
  public final native String get_aa_psipred() /*-{ return this.aa_psipred; }-*/;
  public final native int get_ss_end_pos() /*-{ return this.ss_end_pos; }-*/;
  public final native int get_ss_end_pos_index() /*-{ return this.ss_end_pos_index; }-*/;
  
  public final native String get_hmm_id() /*-{ return this.hmm_id; }-*/;
  public final native int get_pfam_end_pos() /*-{ return this.pfam_end_pos; }-*/;
  public final native int get_pfam_end_pos_index() /*-{ return this.pfam_end_pos_index; }-*/;
  
  public final native Double get_gnomad_af() /*-{ return this.gnomad_af; }-*/;
  public final native String get_gnomad_colorcode() /*-{ return this.gnomad_colorcode; }-*/;
  
  public final native Double get_asa_mean_normalized() /*-{ return this.asa_mean_normalized; }-*/;
  public final native String get_asa_colorcode() /*-{ return this.asa_colorcode; }-*/;
  }


