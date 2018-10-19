package com.alphame.imputation.client;
import java.util.*;
import com.google.gwt.user.client.ui.*;
import com.google.gwt.user.client.ui.FormPanel.SubmitCompleteEvent;

//import jdk.management.resource.internal.inst.WindowsAsynchronousFileChannelImplRMHooks;

import com.google.gwt.core.client.*;
import com.google.gwt.event.dom.client.*;
import com.google.gwt.dom.client.Element;
import com.google.gwt.dom.client.Style.Unit;
import com.google.gwt.http.client.*;
import com.google.gwt.user.client.Window;

public class Imputation implements EntryPoint {	
	  
	  //customized widget
	  private HorizontalPanel TextBoxWithLabel(String label_text, TextBox textbox, int width) {
		HorizontalPanel wrapper = new HorizontalPanel();
		HTML label = new HTML(label_text);
		label.setStyleName("label_upload1");
		wrapper.setSpacing(5);
		wrapper.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		wrapper.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);	 
		wrapper.setStyleName("paddedHorizontalPanel");
		wrapper.add(label);
		wrapper.add(textbox);
		textbox.setHeight("8px");
		textbox.setWidth(width + "px");
		
		return wrapper;
	  }
	
	  //customized widget
	  private HorizontalPanel ConcateWidgets(Widget widget1, Widget widget2) {
		HorizontalPanel wrapper = new HorizontalPanel();
		wrapper.setSpacing(5);
		wrapper.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		wrapper.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);	 
//		wrapper.setStyleName("paddedHorizontalPanel");
		wrapper.add(widget1);
		wrapper.add(widget2);		
		return wrapper;
	  }
	  
	  
	  //resource
	  ImputationResource impRes = GWT.create(ImputationResource.class);
	  
	  //window resolution
	  int s_height = Window.getClientHeight();
	  int s_width = Window.getClientWidth();

	  Double panel_up_ratio = 0.08;
	  Double panel_left_ratio = 0.2;
	  Double panel_up_right_ratio = 0.08;
	  Double panel_down_right_ratio = 0.82;
	  Double panel_left_up_right_ratio = 0.45;
	  Double panel_legend_up_right_ratio = 0.2;

	  //CGI/WSGI JSON related
//	  public String SERVER_URL = "http://35.231.183.217/";
	  public String SERVER_URL = "http://impute.varianteffect.org/";
	  public String JSON_URL = SERVER_URL + "wsgi/imputation_wsgi.wsgi?";
	  
	  public String Jochen_Download_URL = SERVER_URL + "downloads/Weile2017.zip";
	  public String template_Download_URL = SERVER_URL + "downloads/imputation_template.zip";
	  public String example_Download_URL = SERVER_URL + "downloads/imputation_example.zip";
	  public String uniprot_list_Download_URL = SERVER_URL + "downloads/supported_uniprot_ids.txt";
	  public String help_Download_URL = SERVER_URL + "downloads/imputation_help.pdf";
	  public int jsonRequestId = 0;
	  public int intRequest; 
	  public int if_normalization;
	  public int if_regularization;
	  public int if_rawprocessing;
	  public int if_dataquality;
	  public int if_auto_trainquality;
	  public String session_id = "";
	  public String upload_landscape_filename = "";
	  public String upload_fasta_filename = "";
	  public String uniprot_id = "NA";
	  public String regression_cutoff = "0";
	  public String data_cutoff = "0";
	  public String synstop_cutoff = "0";
	  public String stop_exclusion = "0";
	  public String proper_count = "6";
	  public String email_address = "";
	  
	  //panels
	  //private SplitLayoutPanel panel_rootsplit = new SplitLayoutPanel();
	  //private DockPanel panel_rootsplit = new DockPanel();
	  private DockLayoutPanel panel_rootsplit = new DockLayoutPanel(Unit.PX);
	  private DockLayoutPanel panel_up = new DockLayoutPanel(Unit.PX);
	  //private DockLayoutPanel panel_left = new DockLayoutPanel(Unit.PX);
	  private DockLayoutPanel panel_right = new DockLayoutPanel(Unit.PX);
	  private DockLayoutPanel panel_up_right = new DockLayoutPanel(Unit.PX);
	  private DockLayoutPanel panel_down_right = new DockLayoutPanel(Unit.PX);
	  private ScrollPanel panel_down_right_scroll = new ScrollPanel();
	  private HorizontalPanel panel_left_up_right = new HorizontalPanel();
	  private HorizontalPanel panel_middle_up_right = new HorizontalPanel();
	  private DockLayoutPanel panel_right_up_right = new DockLayoutPanel(Unit.PX);

	  private LayoutPanel panel_image1_right = new LayoutPanel();
	  private LayoutPanel panel_image2_right = new LayoutPanel();
	  private DockLayoutPanel panel_image_right = new DockLayoutPanel(Unit.PX);
	  
	  private DockLayoutPanel panel_left_up = new DockLayoutPanel(Unit.PX);
	  private PopupPanel ChartPopupPanel = new PopupPanel();
	  private PopupPanel ErrorPopupPanel = new PopupPanel();
	  private PopupPanel HelpPopupPanel = new PopupPanel();
	  private PopupPanel HelpUploadPopupPanel = new PopupPanel();
	  private PopupPanel HelpViewPopupPanel = new PopupPanel();
	  private DockLayoutPanel PopVeticalPanel = new DockLayoutPanel(Unit.PX);
	  
	  private StackLayoutPanel panel_left = new StackLayoutPanel(Unit.PX);
	  private FormPanel panel_form_left = new FormPanel();
	  private VerticalPanel panel_search_left = new VerticalPanel();
	  private VerticalPanel panel_upload_form_left = new VerticalPanel();
	  private VerticalPanel panel_upload_form_left_options = new VerticalPanel();
	  private VerticalPanel panel_downloads_left = new VerticalPanel();
	  
	  
	  private DockLayoutPanel ErrorLayoutPanel = new DockLayoutPanel(Unit.PX);
	  private VerticalPanel ErrorOkPanel = new VerticalPanel();
	  private VerticalPanel ErrorUpPanel = new VerticalPanel();

	  
	  private VerticalPanel panel_right_up = new VerticalPanel();
	  private HorizontalPanel panel_middle_up = new HorizontalPanel();
	  
	  private DockLayoutPanel panel_help_layout = new DockLayoutPanel(Unit.PX);
	  private VerticalPanel panel_help_close = new VerticalPanel();
	  private HTMLPanel panel_help_html = new HTMLPanel(impRes.help_html().getText());
	  private ScrollPanel panel_help_scroll = new ScrollPanel();
	  
	  private HorizontalPanel img_loading_panel = new HorizontalPanel();
	  
	  //FlexTables
	  private MyFlexTable LandscapePopHitTable = new MyFlexTable();
	    
	  //Booleans
	  public boolean blnFixPopUp = false;
	  public boolean blnPopUp = false;
	  public boolean blnUploadHelpPopUp = false;
	  public boolean blnViewHelpPopUp = false;
	  
	  //ListBox
	  private ListBox lsb_viewoptions = new ListBox();
	  private ListBox lsb_avaliable_landscapes = new ListBox();
	  
	  //CheckBox	  
	  private CheckBox cb_rawprocessing = new CheckBox("Raw data");
	  private CheckBox cb_normalization = new CheckBox("Data rescaling");
	  private CheckBox cb_regularization = new CheckBox("Standard deviation regularization");
	  private CheckBox cb_dataquality = new CheckBox("Fliter low quality variants");
	  private CheckBox cb_autoquality = new CheckBox("Auto training variant quality cutoff");

	  //Button
	  private Button btn_view_preimputed_maps = new Button("View");
	  private Button btn_view_session_maps = new Button("View");
	  private Button btn_view_uniprotid_maps = new Button("View");
	  private Button btn_upload= new Button("Impute");	  
	  private Button btn_download_csv= new Button("Download CSV");
	  private Button btn_download_figure= new Button("Download Figure");
	  private Button btn_pubmed_link = new Button("PubMed Link");
	  private Button btn_download_Jochen= new Button("Download Weile et,.al 2017");
	  private Button btn_download_template= new Button("Download Data Template");
	  private Button btn_download_example= new Button("Download Example - UBE2I");
	  private Button btn_download_uniprot_list= new Button("Download Supported Uniprot IDs");
	  private Button btn_download_help= new Button("Download Help Document");
	  private Button btn_error_ok = new Button("OK");
	  private Button btn_help_exit = new Button("CLOSE");
//	  private Button btn_img_select1= new Button("Original Landscape");
//	  private Button btn_img_select2= new Button("Imputed Landscape");
//	  private Button btn_img_select3= new Button("Polyphen Landscape");
//	  private Button btn_img_select4= new Button("Provean Landscape");
//	  private Button btn_img_select5= new Button("Initial abundance Landscape");
	    
	  //upload
	  private FileUpload upload_landscape = new FileUpload();
	  private FileUpload upload_fasta = new FileUpload();
			  
	  //images
	  private Image img_loading = new Image(impRes.loading()); 
	  private Image img_logo = new Image(impRes.logo());
	  private Image img_imputation_legend = new Image(impRes.imputation_legend());
	  private Image img_provean_legend = new Image(impRes.provean_legend());
	  private Image img_polyphen_legend = new Image(impRes.polyphen_legend());
	  private Image img_sift_legend = new Image(impRes.sift_legend());
	  private Image img_gnomad_legend = new Image(impRes.gnomad_legend());
	  private Image img_help = new Image(impRes.help());
	  private Image img_help_view_landscape = new Image(impRes.question_mark());
	  private Image img_help_upload = new Image(impRes.question_mark());
	  private Image img_example = new Image(impRes.example());
	  private Image img_help_downloads = new Image(impRes.question_mark());
	  private Image img_error = new Image(impRes.error());
	  private Image img_upload_help = new Image(impRes.upload_help());
	  private Image img_view_help = new Image(impRes.view_help());

	  //labels
	  private Label lbl_error = new HTML(); 
	  private Label lbl_error_detail = new HTML();
	  private Label lbl_upload_select1 = new Label("(1) Select Landscape File");
	  private Label lbl_upload_select2 = new Label("(2) Select Protein Fasta File");
	  private Label lbl_upload_select3 = new Label("(3) Input Uniprot ID");
	  private Label lbl_upload_select4 = new Label("(4) Set Parameters");
	  private Label lbl_upload_select5 = new Label("(5) Name Your Session");
	  private Label lbl_upload_select6 = new Label("(6) Email Address");
//	  private Label lbl_upload_select7 = new Label("(7) ");
	  
	  private Label lbl_view_select1 = new Label("(1) Previously imputed maps");
	  private Label lbl_view_select2 = new Label("(2) View maps by session ID");
	  private Label lbl_view_select3 = new Label("(3) View maps by Uniprot ID");
	  
	  private Label lbl_title  = new Label("Human Protein Variant Effect Map Imputation Toolkit");
	  private Label lbl_protein_desc = new Label("");
	  private Label lbl_view_options = new Label("View Options");
	  private Label lbl_available_lanscapes = new Label("Avaliable Landscapes");

	  
	  private HTML PopupLabel = new HTML("");	
	  private HTML PopupLabel1 = new HTML("");	


	  //texts
	  private TextBox txt_querflag_upload = new TextBox();  
	  private TextBox txt_imageheight_upload = new TextBox();
	  private TextBox txt_imagewidth_upload = new TextBox();
	  private TextBox txt_training_cutoff_upload = new TextBox();
	  private TextBox txt_data_cutoff_upload = new TextBox();
	  private TextBox txt_synstop_cutoff_upload = new TextBox();
	  private TextBox txt_stop_exclusion_upload = new TextBox();
	  private TextBox txt_proper_count_upload = new TextBox();
	  private HorizontalPanel txt_synstop_cutoff  = TextBoxWithLabel("• syn/nonsense variant quality cutoff:",txt_synstop_cutoff_upload,40);
	  private HorizontalPanel txt_stop_exclusion  = TextBoxWithLabel("• nonsense variant exclusion regions:",txt_stop_exclusion_upload,80);
	  private HorizontalPanel txt_proper_count  = TextBoxWithLabel("• minimum number of replicates to skip:",txt_proper_count_upload,40);
	  private HorizontalPanel txt_data_cutoff  = TextBoxWithLabel("• variant quality cutoff:",txt_data_cutoff_upload,40);
	  private HorizontalPanel txt_training_cutoff  = TextBoxWithLabel("• training variant quality cutoff:",txt_training_cutoff_upload,40);
	  private TextBox txt_uniprotid_upload = new TextBox();
	  private TextBox txt_name_session = new TextBox();
	  private TextBox txt_sessionid_upload = new TextBox();
	  
	  private TextBox txt_email_upload = new TextBox();
	  private TextBox txt_callback_upload = new TextBox();
	  private TextArea txt_debug = new TextArea();
	  
	  private TextBox txt_uniprotid_view = new TextBox();
	  private TextArea txt_sessionid_view = new TextArea();
	  
	  private TextBox txt_error_email = new TextBox();
	  
	  //ArrayList	  
	  private ArrayList<ImputationData> lstImputationData = new ArrayList<ImputationData>();
	  private ArrayList<AvailableData> lstAvailableData = new ArrayList<AvailableData>();
	  private ArrayList<ErrorData> lstErrorData = new ArrayList<ErrorData>();
	  private ArrayList<SingleLineData> lstSingleLineData = new ArrayList<SingleLineData>();
	  
	  //JArray
	  private final native JsArray<ImputationData> asArrayOfImputationData(JavaScriptObject jso) /*-{ return jso; }-*/;
	  private final native JsArray<AvailableData> asArrayOfAvailableData(JavaScriptObject jso) /*-{ return jso; }-*/;
	  private final native JsArray<ErrorData> asArrayOfErrorData(JavaScriptObject jso) /*-{ return jso; }-*/;
	  private final native JsArray<SingleLineData> asArrayOfSingleLineData(JavaScriptObject jso) /*-{ return jso; }-*/;
	  
	  //FlexTable
	  private FlexTableWithMouseEvent FlexTable_Landscape = new FlexTableWithMouseEvent();
	  private FlexTable HmmHitChartTableFix = new FlexTable();
	  
	  //customized widgets
	  private HorizontalPanel ErrorEmailPanel = ConcateWidgets(new HTML("Your Email"),txt_error_email);
	  
	  public static void doGet(String url) {
		    RequestBuilder builder = new RequestBuilder(RequestBuilder.GET, url);

		    try {
		      Request response = builder.sendRequest(null, new RequestCallback() {
		        public void onError(Request request, Throwable exception) {
		          // Code omitted for clarity
	        			Window.alert("request error!");
		        }

		        public void onResponseReceived(Request request, Response response) {
		        		Window.alert("on response received: (" + response.getText() + ")");
		        }
		      });
		    } catch (RequestException e) {
		      // Code omitted for clarity
        			Window.alert("request exception!");
		    }
		  }
	  
	  public void onModuleLoad() {		  
		  //Window.alert(s_width +"*"+ s_height);
		  //Root panel
		  RootLayoutPanel panel_root = RootLayoutPanel.get();
		  panel_root.add(panel_rootsplit);		  
		  panel_rootsplit.addNorth(panel_up, s_height*panel_up_ratio);		  
		  panel_rootsplit.addWest(panel_left, s_width*panel_left_ratio);		 
//		  panel_rootsplit.addWest(panel_left, s_width);
		  panel_rootsplit.add(panel_right);
		  
		  if (s_height <=1080) {panel_rootsplit.setStyleName("backgroundImage1080");}
		  if (s_height >1080) {panel_rootsplit.setStyleName("backgroundImage");}
		  panel_left.setStyleName("borderPanel");
		  panel_up_right.setStyleName("borderPanel");
		  panel_down_right.setStyleName("borderPanel");

		  //panel_up
		  img_logo.setPixelSize((int) Math.round(s_height*0.06), (int) Math.round(s_height*0.06));
		  panel_up.addWest(panel_left_up,s_width*0.07);
//		  panel_up.addNorth(new HTML(""), s_height*0.03);
//		  panel_up.addSouth(txt_debug, s_height*0.03);
//		  panel_up.addSouth(new HTML(""), s_height*0.03);
		  panel_right_up.getElement().setAttribute("width", ""+s_width*0.15);
		  panel_up.addEast(panel_right_up,s_width*0.15);
		  panel_right_up.setSpacing((int) Math.round(s_height*panel_up_ratio*0.1));
		  panel_right_up.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_RIGHT);
		  panel_right_up.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  
		  
		  btn_download_Jochen.setHeight(s_height*panel_up_ratio*0.4 + "px");
		  btn_download_template.setHeight(s_height*panel_up_ratio*0.4 + "px");
		  btn_download_example.setHeight(s_height*panel_up_ratio*0.4 + "px");
		  btn_download_uniprot_list.setHeight(s_height*panel_up_ratio*0.4 + "px");
		  btn_download_help.setHeight(s_height*panel_up_ratio*0.4 + "px");		  
		  
		  btn_download_Jochen.setWidth(s_width*0.15 + "px");
		  btn_download_template.setWidth(s_width*0.15 + "px");
		  btn_download_example.setWidth(s_width*0.15 + "px");
		  btn_download_uniprot_list.setWidth(s_width*0.15 + "px");
		  btn_download_help.setWidth(s_width*0.15 + "px");
		  
//		  panel_right_up.add(btn_download_Jochen);		  
//		  panel_right_up.add(btn_download_template);
		  img_help.setStyleName("hand_cursor");
		  panel_right_up.add(img_help);
				  
		  panel_up.add(panel_middle_up);
		  panel_middle_up.setHeight(s_height*panel_up_ratio +"px");
		  panel_middle_up.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  panel_middle_up.add(lbl_title);
		  

		  panel_left_up.addNorth(new HTML(""), s_height*0.01);
		  panel_left_up.addSouth(new HTML(""), s_height*0.01);
		  panel_left_up.addEast(new HTML(""), s_height*0.01);
		  panel_left_up.addWest(new HTML(""), s_height*0.01);
		  panel_left_up.add(img_logo);

		  //panel_right
		  panel_right.addNorth(panel_up_right,s_height*panel_up_right_ratio);
		  panel_right.add(panel_down_right);
		  
		  //panel_left_up_right
		  panel_up_right.addWest(panel_left_up_right,s_width*(1-panel_left_ratio)*panel_left_up_right_ratio);
		  panel_left_up_right.setSpacing((int) (s_width*0.01));
		  panel_left_up_right.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  panel_left_up_right.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);	  
		  panel_left_up_right.add(btn_download_csv);
		  panel_left_up_right.add(btn_download_figure);
		  panel_left_up_right.add(btn_pubmed_link);
		  panel_left_up_right.add(lbl_view_options);
		  panel_left_up_right.add(lsb_viewoptions);	
		  
//		  img_imputation_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
//		  img_provean_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
//		  img_polyphen_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
//		  img_sift_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
//		  img_gnomad_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));

		  panel_up_right.addEast(panel_right_up_right,s_width*(1-panel_left_ratio)*panel_legend_up_right_ratio);	  
		  
		  //panel_middle_up_right
		  panel_up_right.add(panel_middle_up_right);	
		  panel_middle_up_right.setSpacing((int) (s_width*0.01));
//		  panel_middle_up_right.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
//		  panel_middle_up_right.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_RIGHT);	  
		  panel_middle_up_right.add(new Image(impRes.protein()));	  
		  panel_middle_up_right.add(lbl_protein_desc);
		  lbl_protein_desc.setStyleName("label_protein");

		  //ChartPopupPanel
		  ChartPopupPanel.setWidth(s_width*0.15 +"px");
		  ChartPopupPanel.setHeight(s_height*0.3 +"px");
		  ChartPopupPanel.setStyleName("popup_panel");
		  ChartPopupPanel.setWidget(LandscapePopHitTable);
		  
		  //ErrorPopupPanel
		  ErrorOkPanel.getElement().setAttribute("width", ""+s_width*0.3);
		  ErrorOkPanel.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  ErrorOkPanel.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  ErrorOkPanel.add(btn_error_ok);
		  ErrorOkPanel.setSpacing(5);

		  ErrorUpPanel.getElement().setAttribute("width", ""+s_width*0.3);
//		  ErrorUpPanel.add(img_error);	
//		  ErrorUpPanel.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
//		  ErrorUpPanel.add(lbl_error);

		  ErrorUpPanel.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  ErrorUpPanel.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  ErrorUpPanel.add(ConcateWidgets(img_error,lbl_error));
		  ErrorUpPanel.add(ErrorEmailPanel);
		  ErrorEmailPanel.setVisible(false);
		  
		  ErrorUpPanel.setSpacing(15);
		  
		  ErrorLayoutPanel.addSouth(ErrorOkPanel,s_height*0.05);
		  ErrorLayoutPanel.add(ErrorUpPanel);
		  
		  ErrorPopupPanel.setWidth(s_width*0.3 +"px");
		  ErrorPopupPanel.setHeight(s_height*0.2 +"px");
		  ErrorPopupPanel.setStyleName("popup_panel");
		  ErrorPopupPanel.setWidget(ErrorLayoutPanel);

		  
		  //HelpPopUpPanel		  
		  HelpPopupPanel.setWidth(s_width*0.6 +"px");
		  HelpPopupPanel.setHeight(s_height*0.6 +"px");
		  HelpPopupPanel.setStyleName("popup_panel");
		  
		  panel_help_close.getElement().setAttribute("width", ""+s_width*0.6);
		  panel_help_close.setSpacing(10);
		  panel_help_close.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_help_close.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  panel_help_close.add(btn_help_exit);
		  panel_help_layout.addSouth(panel_help_close, s_height*0.05);
		  panel_help_layout.add(panel_help_scroll);
		  panel_help_scroll.add(panel_help_html);
		  panel_help_scroll.setStyleName("disable_horizontal_scroll");
		  HelpPopupPanel.setWidget(panel_help_layout);
		  
		  //panel_left
		  //panel_left - upload and impute stack
		  FocusPanel Header1 = new FocusPanel(); 
		  Header1 = createHeaderWidgetWithExample("Upload & Impute",new Image(impRes.upload()),img_help_upload,img_example);
//		  panel_left.add(panel_form_left,createHeaderWidget("Upload & Impute",new Image(impRes.upload()),img_help_upload),50);
		  panel_left.add(panel_form_left,Header1,50);
		  panel_form_left.setWidget(panel_upload_form_left);
		  panel_middle_up_right.setVerticalAlignment(HasVerticalAlignment.ALIGN_BOTTOM);
		  panel_upload_form_left.add(lbl_upload_select1);		  
		  panel_upload_form_left.add(upload_landscape);
		  panel_upload_form_left.add(cb_rawprocessing);

		  panel_upload_form_left.add(lbl_upload_select2);
		  panel_upload_form_left.add(upload_fasta);
		  panel_upload_form_left.add(lbl_upload_select3);		  
		  panel_upload_form_left.add(txt_uniprotid_upload);
		  panel_upload_form_left.add(lbl_upload_select4);	
		  panel_upload_form_left.add(panel_upload_form_left_options);
		  
		  
		  panel_upload_form_left_options.add(cb_dataquality);
		  panel_upload_form_left_options.add(txt_data_cutoff);
		  panel_upload_form_left_options.add(cb_normalization);
		  panel_upload_form_left_options.add(txt_synstop_cutoff);
		  panel_upload_form_left_options.add(txt_stop_exclusion);
		  panel_upload_form_left_options.add(cb_regularization);	
		  panel_upload_form_left_options.add(txt_proper_count);
		  panel_upload_form_left_options.add(cb_autoquality);
		  panel_upload_form_left_options.add(txt_training_cutoff);
		  panel_upload_form_left_options.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_upload_form_left_options.getElement().setAttribute("width", ""+s_width*panel_left_ratio);
		  panel_upload_form_left_options.setSpacing(0);
		  
		  panel_upload_form_left.add(lbl_upload_select5);	
		  
		  panel_upload_form_left.add(txt_sessionid_upload);
		  panel_upload_form_left.add(txt_name_session);
		  panel_upload_form_left.add(lbl_upload_select6);	
//		  panel_upload_form_left.add(lbl_upload_select7);	
		  panel_upload_form_left.add(txt_email_upload);		  
		  panel_upload_form_left.add(new HTML(""));
		  panel_upload_form_left.add(new HTML(""));
		  panel_upload_form_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_upload_form_left.add(btn_upload);
		  panel_upload_form_left.add(txt_querflag_upload);
		  panel_upload_form_left.add(txt_imagewidth_upload);
		  panel_upload_form_left.add(txt_imageheight_upload);

		  panel_upload_form_left.add(txt_callback_upload);
		  panel_upload_form_left.getElement().setAttribute("width", ""+s_width*panel_left_ratio);
		  panel_upload_form_left.setSpacing(4);
		  btn_upload.setStyleName("hvr-wobble-top");		  
		  upload_fasta.setStyleName("file_upload");
		  upload_landscape.setStyleName("file_upload");
		  lbl_upload_select1.setStyleName("label_upload");
		  lbl_upload_select2.setStyleName("label_upload");
		  lbl_upload_select3.setStyleName("label_upload");
		  lbl_upload_select4.setStyleName("label_upload");
		  lbl_upload_select5.setStyleName("label_upload");
		  lbl_upload_select6.setStyleName("label_upload");
//		  lbl_upload_select7.setStyleName("label_upload");
		  cb_rawprocessing.setStyleName("checkbox_upload");
		  cb_normalization.setStyleName("checkbox_upload");
		  cb_regularization.setStyleName("checkbox_upload");
		  cb_dataquality.setStyleName("checkbox_upload");
		  cb_autoquality.setStyleName("checkbox_upload");
		  // Upload Action
		  panel_form_left.setAction(JSON_URL);
		  panel_form_left.setEncoding(FormPanel.ENCODING_MULTIPART);
		  panel_form_left.setMethod(FormPanel.METHOD_POST);
		  upload_landscape.setName("upload_landscape");
		  upload_fasta.setName("upload_fasta");
		  txt_querflag_upload.setName("queryflag");
		  txt_querflag_upload.setText("0");
		  txt_querflag_upload.setVisible(false);		  
		  txt_imageheight_upload.setName("imageheight");
		  txt_imageheight_upload.setText(s_height+"");
		  txt_imageheight_upload.setVisible(false);		  
		  txt_imagewidth_upload.setName("imagewidth");
		  txt_imagewidth_upload.setText(s_width+"");
		  txt_imagewidth_upload.setVisible(false);		
		  txt_sessionid_upload.setName("sessionid");
		  txt_sessionid_upload.setText("");
		  txt_sessionid_upload.setVisible(false);	
		  txt_callback_upload.setName("callback");
		  txt_callback_upload.setText("0");
		  txt_callback_upload.setVisible(false);	
		  txt_training_cutoff_upload.setName("regression_cutoff");
		  txt_data_cutoff_upload.setName("data_cutoff");
		  txt_uniprotid_upload.setName("proteinid");
		  cb_normalization.setName("normalization");
		  cb_regularization.setName("regularization");
		  cb_rawprocessing.setName("rawprocessing");
		  cb_autoquality.setName("auto_cutoff");
		  
		  //panel_left - View Landscapes stack
		  btn_view_preimputed_maps.setStyleName("hvr-wobble-top");
		  btn_view_session_maps.setStyleName("hvr-wobble-top");
		  btn_view_uniprotid_maps.setStyleName("hvr-wobble-top");
		  lbl_view_select1.setStyleName("label_upload");
		  lbl_view_select2.setStyleName("label_upload");
		  lbl_view_select3.setStyleName("label_upload");

		  panel_search_left.add(lbl_view_select1);
		  panel_search_left.add(lsb_avaliable_landscapes);
		  panel_search_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_search_left.add(btn_view_preimputed_maps);
		  panel_search_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_LEFT);
		  panel_search_left.add(lbl_view_select2);
		  panel_search_left.add(txt_sessionid_view);
		  txt_sessionid_view.setWidth(s_width*0.12 + "px");
		  panel_search_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_search_left.add(btn_view_session_maps);
		  panel_search_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_LEFT);
		  panel_search_left.add(lbl_view_select3);
		  panel_search_left.add(txt_uniprotid_view);
		  panel_search_left.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  panel_search_left.add(btn_view_uniprotid_maps);
		  
		  panel_search_left.setSpacing(10);
		  FocusPanel Header2 = new FocusPanel();
		  
		  img_help_view_landscape.setStyleName("hand_cursor");
		  img_help_downloads.setStyleName("hand_cursor");
		  img_help_upload.setStyleName("hand_cursor");
		  img_example.setStyleName("hand_cursor");
		  Header2 = createHeaderWidget("View Landscapes",new Image(impRes.view()),img_help_view_landscape);
		  panel_left.add(panel_search_left,Header2,70);

		  //panel_left - Download & help stack
		  panel_downloads_left.add(btn_download_Jochen);
		  panel_downloads_left.add(btn_download_template);
//		  panel_downloads_left.add(btn_download_example);
		  panel_downloads_left.add(btn_download_help);
		  panel_downloads_left.add(btn_download_uniprot_list);
		  panel_downloads_left.setSpacing(15);
		  FocusPanel Header3 = new FocusPanel();
		  Header3 = createHeaderWidget("Downloads",new Image(impRes.miscellaneous()), img_help_downloads);
		  panel_left.add(panel_downloads_left,Header3,70);
		  
		  //lsb_viewoptions
		  lsb_viewoptions.addItem("Original Fitness");
//		  lsb_viewoptions.addItem("Original Fitness - reversed");
		  lsb_viewoptions.addItem("Refined Fitness");
		  lsb_viewoptions.addItem("Polyphen Score");
		  lsb_viewoptions.addItem("SIFT Score");
		  lsb_viewoptions.addItem("Provean Score");
		  lsb_viewoptions.addItem("Allele Frequency");
		  
		  lsb_viewoptions.addChangeHandler(new ChangeHandler() {
			  public void onChange(ChangeEvent event) {
	              show_processing_img();
				  refresh_imputation_table(lsb_viewoptions.getSelectedItemText());
		    }
		  });

		    
		  Header2.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		get_available_landscapes();
			    } 
		  });
		  		  
		  cb_normalization.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		if (cb_normalization.getValue()){
//			    			txt_synstop_cutoff_upload.setText("0");
//			    			txt_stop_exclusion_upload.setText("0");
			    			txt_synstop_cutoff.setVisible(true);
			    			txt_stop_exclusion.setVisible(true);
			    		}
			    		else {
			    			txt_synstop_cutoff.setVisible(false);
			    			txt_stop_exclusion.setVisible(false);
			    		}	
			    } 
		  });
		  
		  cb_regularization.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		if (cb_regularization.getValue()){
//			    			txt_proper_count_upload.setText("6");
			    			txt_proper_count.setVisible(true);
			    		}
			    		else {
			    			txt_proper_count.setVisible(false);
			    		}	
			    } 
		  });		  		  
		  
		  cb_dataquality.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		if (cb_dataquality.getValue()){
//			    			txt_data_cutoff_upload.setText("0");
			    			txt_data_cutoff.setVisible(true);
			    		}
			    		else {
//			    			txt_data_cutoff_upload.setText("0");
			    			txt_data_cutoff.setVisible(false);
			    		}	
			    } 
		  });
		  		  
		  cb_autoquality.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		if (cb_autoquality.getValue()){
//			    			txt_training_cutoff_upload.setText("0");
			    			txt_training_cutoff.setVisible(false);
			    		}
			    		else {
//			    			txt_training_cutoff_upload.setText("0");
			    			txt_training_cutoff.setVisible(true);
			    		}	
			    } 
		  });
		  		  
		  img_example.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {			        
			    		set_default_parameters();	
			    		txt_uniprotid_upload.setText("P63279");
			    		txt_name_session.setText("UBE2I_Example");
			        Window.open(example_Download_URL, "_blank", null);
			        HelpPopupPanel.hide();
			        HelpPopupPanel.center();
					HelpPopupPanel.show();					 
					panel_help_html.getElementById("_Toc520809764").scrollIntoView();
					panel_help_scroll.setVerticalScrollPosition(panel_help_scroll.getVerticalScrollPosition() + (int) Math.round(s_height*0.55) - 2*panel_help_html.getElementById("_Toc520809764").getOffsetHeight());
			    } 
		  });
		  
		  img_help.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {	
					  HelpPopupPanel.hide();
					  HelpPopupPanel.center();
					  HelpPopupPanel.show();					 
//					  panel_help_html.getElementById("_Toc518991758").scrollIntoView();
//					  panel_help_scroll.setVerticalScrollPosition(panel_help_scroll.getVerticalScrollPosition() + (int) Math.round(s_height*0.6) - panel_help_html.getElementById("_Toc518991758").getOffsetHeight());
			    } 
		  });
		  
		  img_help_upload.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {	
					  HelpPopupPanel.hide();
					  HelpPopupPanel.center();
					  HelpPopupPanel.show();					 
					  panel_help_html.getElementById("_Toc520809761").scrollIntoView();
					  panel_help_scroll.setVerticalScrollPosition(panel_help_scroll.getVerticalScrollPosition() + (int) Math.round(s_height*0.55) - 2*panel_help_html.getElementById("_Toc520809761").getOffsetHeight());
			    } 
		  });
		  
		  img_help_downloads.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
					  HelpPopupPanel.hide();
					  HelpPopupPanel.center();
					  HelpPopupPanel.show();					 
					  panel_help_html.getElementById("_Toc520809777").scrollIntoView();
					  panel_help_scroll.setVerticalScrollPosition(panel_help_scroll.getVerticalScrollPosition() + (int) Math.round(s_height*0.55) - 2*panel_help_html.getElementById("_Toc520809777").getOffsetHeight());
			    } 
		  });
		  
		  img_help_view_landscape.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {	
					  HelpPopupPanel.hide();
					  HelpPopupPanel.center();
					  HelpPopupPanel.show();					 
					  panel_help_html.getElementById("_Toc520809774").scrollIntoView();
					  panel_help_scroll.setVerticalScrollPosition(panel_help_scroll.getVerticalScrollPosition() + (int) Math.round(s_height*0.55) - 2*panel_help_html.getElementById("_Toc520809774").getOffsetHeight());
			    } 
		  });
		  
		  btn_help_exit.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
					  HelpPopupPanel.hide();	
			    }
		  });
		  
		  btn_error_ok.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    	
//			    		Window.alert(ErrorEmailPanel.isVisible() + "");
			    		if (txt_error_email.getText() != "" && ErrorEmailPanel.isVisible()){
			    			send_error_email(txt_error_email.getText());			    		
			    		}
			    		ErrorEmailPanel.setVisible(false);
		    			ErrorPopupPanel.hide();
			    } 
		  });
		  
		  btn_download_csv.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(SERVER_URL + "output/" + session_id + "_imputation.csv" , "_blank", null);
			    } 
		  });
		  
		  btn_pubmed_link.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			    		open_pubmed_link();
			    } 
		  });
		  		  
		  btn_download_figure.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(SERVER_URL + "output/" + session_id + "_fitness_org.pdf" , "_blank", null);
			        Window.open(SERVER_URL + "output/" + session_id + "_fitness_refine.pdf" , "_blank", null);
			    } 
		  });
		  
		  btn_download_uniprot_list.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(uniprot_list_Download_URL, "_blank", null);
			    } 
		  });
		  
		  btn_download_example.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(example_Download_URL, "_blank", null);
			    } 
		  });
		  
		  btn_download_help.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(help_Download_URL, "_blank", null);
			    } 
		  });
		  		  		  
		  btn_download_Jochen.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open("http://dalai.mshri.on.ca/~jweile/projects/dmsData/", "_blank", null);
			    } 
		  });
		  
		  btn_download_template.addClickHandler(new ClickHandler() {
			    @Override
			    public void onClick(ClickEvent event) {
			        Window.open(template_Download_URL, "_blank", null);
			    } 
		  });
		  	
		  lsb_avaliable_landscapes.addChangeHandler(new ChangeHandler() {
			  public void onChange(ChangeEvent event) {
//	              show_processing_img();
//	              String cur_id =  lsb_avaliable_landscapes.getSelectedItemText();
//            	  lbl_protein_desc.setText(cur_id);
//				  uniprot_id = cur_id.substring(0,6);
//				  session_id = cur_id.substring(7,cur_id.length()-1);				  
////				  Window.alert(uniprot_id + "|" + session_id);
//				  get_available_landscape_data(cur_id);
//				  panel_down_right_scroll.scrollToLeft();
				  
		    }
		  });
		  
		  btn_view_preimputed_maps.addClickHandler(new ClickHandler() {
		         @Override
		         public void onClick(ClickEvent event) {		        	 	 
		        	 	  if (img_loading.isVisible()){
		        	 		  return;
		        	 	  }
		              show_processing_img();
		              session_id =  lsb_avaliable_landscapes.getSelectedItemText();
	            	  	  lbl_protein_desc.setText(session_id);
//					  uniprot_id = cur_id.substring(0,6);
//					  session_id = cur_id.substring(7,cur_id.length()-1);				  
//					  Window.alert(uniprot_id + "|" + session_id);
					  get_available_landscape_data(session_id);
					  panel_down_right_scroll.scrollToLeft();			        
	            	}	
		      });
		  		  
		  btn_view_session_maps.addClickHandler(new ClickHandler() {
		         @Override
		         public void onClick(ClickEvent event) {
		        	 	  if (img_loading.isVisible()){
		        	 		  return;
		        	 	  }
		              show_processing_img();
		              session_id =  txt_sessionid_view.getText();
	            	  lbl_protein_desc.setText(session_id);
					  get_available_landscape_data(session_id);
					  panel_down_right_scroll.scrollToLeft();			        
	            	}	
		      });
		  
		  btn_view_uniprotid_maps.addClickHandler(new ClickHandler() {
		         @Override
		         public void onClick(ClickEvent event) {
		        	 	  if (img_loading.isVisible()){
		        	 		  return;
		        	 	  }
		              show_processing_img();
		              session_id =  "!" + txt_uniprotid_view.getText() ;
	            	  lbl_protein_desc.setText(session_id);
					  get_available_landscape_data(session_id);
					  panel_down_right_scroll.scrollToLeft();			        
	            	}	
		      });
		  		  
		  btn_upload.addClickHandler(new ClickHandler() {
		         @Override
		         public void onClick(ClickEvent event) {
		        	 
		        	 
//	        	 	 	Window.alert("TEST");
//	        	 	 	Window.alert(img_loading.isAttached() +"");
		        	    if (img_loading.isVisible()) {
		        	    		return;
		        	    	}
			        	Random rand = new Random();
			        	Date date = new Date();
			        	String[] date_splitted  = date.toString().split(" ");
			        	String date_concated = "";
			        	for (int i = 0; i < 4; i++) {
//			        		Window.alert(date_splitted[i]);
			        		date_concated += date_splitted[i] + "_";
				        	date_concated = date_concated.replaceAll(":", "_");
			        	}
			        	
			        int can_submit = 1;
			        
			        if (upload_landscape.getFilename().length() == 0 ) {
//			        		Window.alert("No landscape file specified!");
				    	    lbl_error.setText("No lanscape file specified!");
				    		ErrorPopupPanel.center();
			        		can_submit = 0; 
			        		return;
			        	} 
		            if (upload_fasta.getFilename().length() == 0 ) {
//		            		Window.alert("No fasta file specified!");
				    	    lbl_error.setText("No fasta file specified!");
				    		ErrorPopupPanel.center();
		            		can_submit = 0; 
		            		return;
		            	}
		            if (txt_uniprotid_upload.getText().length() != 6 ) {
//		            		Window.alert("Please input correct Uniprot ID!"); 
				    	    lbl_error.setText("Please input correct Uniprot ID!");
				    		ErrorPopupPanel.center();
		            		can_submit = 0; 
		            		return;
		            	}
		            if (!txt_training_cutoff_upload.getText().matches("-?\\d+(\\.\\d+)?") && !cb_autoquality.getValue()) {
//		            		Window.alert("Please input correct cutoff!");
				    	    lbl_error.setText("Please input correct training quality cutoff!");
				    		ErrorPopupPanel.center();
		            		can_submit = 0; 
		            		return;
		            	}		            		            		            
		            if (!txt_data_cutoff_upload.getText().matches("-?\\d+(\\.\\d+)?") && cb_dataquality.getValue()) {
	//	            		Window.alert("Please input correct cutoff!");
				    	    lbl_error.setText("Please input correct data quanlity cutoff!");
				    		ErrorPopupPanel.center();
		            		can_submit = 0; 
		            		return;
	            		}	
		            if (!txt_synstop_cutoff_upload.getText().matches("-?\\d+(\\.\\d+)?") && cb_normalization.getValue()) {
	//	            		Window.alert("Please input correct cutoff!");
				    	    lbl_error.setText("Please input correct data quanlity cutoff!");
				    		ErrorPopupPanel.center();
		            		can_submit = 0; 
		            		return;
	            		}
		            
		            if (can_submit == 1){
		               //submit the form
					    uniprot_id = txt_uniprotid_upload.getText();
					    regression_cutoff = txt_training_cutoff_upload.getText();
					    email_address = txt_email_upload.getText();					    		
					    data_cutoff = txt_data_cutoff_upload.getText();
		            		session_id = txt_name_session.getText();
		            		if (txt_name_session.getText() == "") {
		            			session_id = uniprot_id + "[" + date_concated + String.valueOf(rand.nextInt(50)) + "]";	
		            		}else {
		            			session_id = uniprot_id + "[" + txt_name_session.getText() + "]";
		            		}
			        	 	if (event.isShiftKeyDown()){
			        	 		session_id = "*" + session_id;
			        	 	}
			        	 	
			        	 	txt_sessionid_upload.setText(session_id);

			            upload_landscape_filename = upload_landscape.getFilename();
			            upload_fasta_filename = upload_fasta.getFilename();
					    show_processing_img();
					    
					    //rescaling
					    if (cb_normalization.getValue()) { 
					    		if_normalization = 1;
					    		synstop_cutoff = txt_synstop_cutoff_upload.getText();
					    		stop_exclusion = txt_stop_exclusion_upload.getText();
					    	} 
					    else { 
					    		if_normalization = 0;
					    		synstop_cutoff = "-inf";
					    		stop_exclusion = "0";
					    	}
					    
					    //regularization
					    if (cb_regularization.getValue()) { 
					    		if_regularization = 1;
					    		proper_count = txt_proper_count_upload.getText();
					    }
					    else { 
					    		if_regularization = 0;
					    		proper_count = "8";
					    	}
					    
					    //rawdata? 
					    if (cb_rawprocessing.getValue()) { 
					    		if_rawprocessing = 1;
					    	} else {
					    		if_rawprocessing = 0;
					    	}
					    
					    //filter low quality missense variants
					    if (cb_dataquality.getValue()) { 
					    		if_dataquality = 1;
					    } else {
					    		if_dataquality = 0;
				    		}
					    
					    //auto training quality cutoff
					    if (cb_autoquality.getValue()) { 
					    		if_auto_trainquality = 1;
					    	} else { 
					    		if_auto_trainquality = 0;
					    		regression_cutoff = txt_training_cutoff_upload.getText();
					    	}

					    lbl_protein_desc.setText(session_id);
//					    Window.alert("Ready to submit!");
					    panel_form_left.submit();	
						btn_download_csv.setEnabled(false);
						btn_download_figure.setEnabled(false);
						btn_pubmed_link.setEnabled(false);
		            }				
		         }
		      });
		   
		  panel_form_left.addSubmitCompleteHandler(new FormPanel.SubmitCompleteHandler() {
				@Override
				public void onSubmitComplete(SubmitCompleteEvent event) {
					// TODO Auto-generated method stub
//					Window.alert(event.getResults());		
//					Window.alert("Upload successfully!");
		            get_imputation_data();
		            //btn_img_select1.click();
				}
		      });
	  
		  lbl_title.setStyleName("label_title");
		 		  
		  //test error popup panel
	  	  lbl_error.setStyleName("label_error");

		  btn_download_csv.setEnabled(false);
		  btn_download_figure.setEnabled(false);
		  btn_pubmed_link.setEnabled(false);
		  
		  send_resolution();
		  img_loading.setVisible(false);
		  set_default_parameters();  
	  }
	  
	  private void set_default_parameters() {
		  cb_normalization.setValue(true);
		  txt_synstop_cutoff_upload.setText("0");
		  txt_stop_exclusion_upload.setText("0");
		  txt_synstop_cutoff.setVisible(true);
		  txt_stop_exclusion.setVisible(true);
		  
		  cb_regularization.setValue(true);
		  txt_proper_count_upload.setText("8");
		  txt_proper_count.setVisible(true);
		  
		  cb_dataquality.setValue(true);
		  txt_data_cutoff_upload.setText("0");
		  txt_data_cutoff.setVisible(true);
		  
		  cb_autoquality.setValue(true);
		  txt_training_cutoff_upload.setText("0");
		  txt_training_cutoff.setVisible(false);
		  
	  }
	  
	  private FocusPanel createHeaderWidget(String text, Image image, Image help_image) {
		    // Add the image and text to a horizontal panel
		    FocusPanel wrapper = new FocusPanel();
//		    DockLayoutPanel dPanel = new DockLayoutPanel(Unit.PX);
		    //dPanel.setStyleName("borderPanel");
		    HorizontalPanel hPanel = new HorizontalPanel();
		    hPanel.setHeight("100%");
		    hPanel.setSpacing(10);
		    hPanel.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		    hPanel.add(image);
		    
		    Label headerText = new Label(text);
		    headerText.setStyleName("label_stack_header");
		    //headerText.setStyleName("cw-StackPanelHeader");
		    hPanel.add(headerText);
		    hPanel.add(help_image);
//		    dPanel.add(hPanel);
		    wrapper.add(hPanel);
		    return wrapper;
		  }
	  
	  private FocusPanel createHeaderWidgetWithExample(String text, Image image, Image help_image, Image example_image) {
		    // Add the image and text to a horizontal panel
		    FocusPanel wrapper = new FocusPanel();
//		    DockLayoutPanel dPanel = new DockLayoutPanel(Unit.PX);
		    //dPanel.setStyleName("borderPanel");
		    HorizontalPanel hPanel = new HorizontalPanel();
		    hPanel.setHeight("100%");
		    hPanel.setSpacing(10);
		    hPanel.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		    hPanel.add(image);
		    
		    Label headerText = new Label(text);
		    headerText.setStyleName("label_stack_header");
		    //headerText.setStyleName("cw-StackPanelHeader");
		    hPanel.add(headerText);
		    hPanel.add(help_image);
		    hPanel.add(example_image);
//		    dPanel.add(hPanel);
		    wrapper.add(hPanel);
		    return wrapper;
		  }	  
	  	  
	  private void show_processing_img(){
		  panel_down_right.clear();
		  img_loading_panel.getElement().setAttribute("width", ""+ s_width*(1-panel_left_ratio));
		  img_loading_panel.getElement().setAttribute("height", ""+ s_height*panel_down_right_ratio);
		  img_loading_panel.setVerticalAlignment(HasVerticalAlignment.ALIGN_MIDDLE);
		  img_loading_panel.setHorizontalAlignment(HasHorizontalAlignment.ALIGN_CENTER);
		  img_loading_panel.add(img_loading);
		  img_loading_panel.setSpacing(5);
		  
//		  panel_down_right.addNorth(new HTML(""), (s_height*(panel_down_right_ratio-0.4)/2));
//		  panel_down_right.addWest(new HTML(""), (s_width*(1-panel_left_ratio-0.4)/2));
//		  panel_down_right.addEast(new HTML(""), (s_width*(1-panel_left_ratio-0.4)/2));
//		  panel_down_right.addSouth(new HTML(""), (s_height*(panel_down_right_ratio-0.4)/2));
		  img_loading.setVisible(true);
		  panel_down_right.add(img_loading_panel);
	  }

	  private void refresh_available_landscapes(){
		  lsb_avaliable_landscapes.clear();
		  for (int i = 0; i < lstAvailableData.size(); i++){
			  lsb_avaliable_landscapes.addItem(lstAvailableData.get(i).get_landscape_name());
		  }
	  }
	  
	  private void send_resolution() {		  
    	  String url;
    	  url= JSON_URL;
    	  	  url = URL.encode(url)+ "queryflag=-1;imagewidth=" + s_width + ";imageheight=" + s_height  + ";callback=";
    	  	  txt_debug.setText(url);
//    	  	  Window.alert(url);
	      getJson(jsonRequestId++, url, -1,this);
	  }
	  
	  private void get_imputation_data() {		  
    	  String url;
    	  url= JSON_URL;
    	  	  url = URL.encode(url)+ "queryflag=1;email_address=" + email_address + ";sessionid=" + session_id  + ";proteinid=" + uniprot_id + ";proper_count=" + proper_count + ";stop_exclusion=" + stop_exclusion + ";synstop_cutoff=" + synstop_cutoff + ";data_cutoff=" + data_cutoff + ";regression_cutoff=" + regression_cutoff + ";if_normalization=" + if_normalization + ";if_regularization=" + if_regularization + ";if_rawprocessing=" + if_rawprocessing + ";if_auto_cutoff=" + if_auto_trainquality + ";if_data_cutoff=" + if_dataquality + ";callback=";
    	  	  txt_debug.setText(url);
//    	  	  Window.alert(url);
	      getJson(jsonRequestId++, url, 1,this);
	  }
	  
	  private void get_available_landscapes(){		  
    	  String url;
    	  url= JSON_URL;
	      url = URL.encode(url)+ "queryflag=2;callback=";
	      getJson(jsonRequestId++, url, 2,this);
	  } 
	  
	  private void get_available_landscape_data(String input_session_id){		  
    	  String url;
    	  url= JSON_URL;
	      url = URL.encode(url)+ "queryflag=3;sessionid=" + input_session_id +";callback=";
	      getJson(jsonRequestId++, url, 3,this);
	  }
	  
	  private void open_pubmed_link() {
    	  String url;
    	  url= JSON_URL;
	      url = URL.encode(url)+ "queryflag=4;sessionid=" + session_id +";callback=";
//	      Window.alert(url);
	      getJson(jsonRequestId++, url, 4,this);
	  }  
	  
	  private void send_error_email(String email) {
    	  String url;
    	  url= JSON_URL;
	      url = URL.encode(url)+ "queryflag=5;email_address=" + email + ";sessionid=" + session_id +";callback=";
//	      Window.alert(url);
	      getJson(jsonRequestId++, url, 5,this);  
	  }

	  public native static void getJson(int requestId, String url, int RequestFlag, Imputation handler) /*-{
	   var callback = "callback" + requestId;
	
	   // [1] Create a script element.
	   var script = document.createElement("script");
	   script.setAttribute("src", url+callback);
	   script.setAttribute("type", "text/javascript");
	
	   // [2] Define the callback function on the window object.
	   window[callback] = function(jsonObj) {
	   // [3]
	     handler.@com.alphame.imputation.client.Imputation::handleJsonResponse(Lcom/google/gwt/core/client/JavaScriptObject;I)(jsonObj,RequestFlag);
	     window[callback + "done"] = true;
	   }
	
	   // [4] JSON download has 1-second timeout.
	   setTimeout(function() {
	     if (!window[callback + "done"]) {
	       handler.@com.alphame.imputation.client.Imputation::handleJsonResponse(Lcom/google/gwt/core/client/JavaScriptObject;I)(null,RequestFlag);
	     }
	
	     // [5] Cleanup. Remove script and callback elements.
	     document.body.removeChild(script);
	     delete window[callback];
	     delete window[callback + "done"];	
	   }, 400000);
	
	   // [6] Attach the script element to the document body.
	   document.body.appendChild(script);
	  }-*/;	
	  
	  public void handleJsonResponse(JavaScriptObject jso,int RequestFlag) {
		    if (jso == null) {
//		    		Window.alert("Couldn't retrieve JSON");
		    		return;
		    }  		
		    
		    
		    if (RequestFlag != 5) {
			    if (asArrayOfErrorData (jso).get(0).get_error() != null) {
//			    		Window.alert(asArrayOfErrorData (jso).get(0).get_error());
			    		lbl_error.setText(asArrayOfErrorData (jso).get(0).get_error());
			    	    ErrorEmailPanel.setVisible(true);
			    		ErrorPopupPanel.center();
			    		img_loading.setVisible(false); 
			    		return;
			    	}  			
		    }
		    //	Window.alert("queryflag:" + RequestFlag + ",Json Response Back!");
		    switch (RequestFlag) {
		    case -1:{
				 if (asArrayOfSingleLineData (jso).get(0).get_content() != "OK") {
					lbl_error.setText(asArrayOfSingleLineData (jso).get(0).get_content());
//				    ErrorEmailPanel.setVisible(true);
					ErrorPopupPanel.center();
				}	
				break;
		    }
		    
		    	case 1:{
//		    		 Window.alert(asArrayOfSingleLineData (jso).get(0).get_content());
		    		 if (asArrayOfSingleLineData (jso).get(0).get_content() != null) {
			    		lbl_error.setText(asArrayOfSingleLineData (jso).get(0).get_content());
			    	    ErrorEmailPanel.setVisible(true);
			    		ErrorPopupPanel.center();
			    		img_loading.setVisible(false);
		    		 }else {
			    		lstImputationData.clear();
			    		for (int i = 0; i < asArrayOfImputationData (jso).length(); i++){
			    			lstImputationData.add (asArrayOfImputationData (jso).get(i));
			    		}
		    			lsb_viewoptions.setItemSelected(0,true);
		    			refresh_imputation_table("Original Fitness");
		    		}
	    			break;
	    		}
		    	case 2:{
		    		lstAvailableData.clear();
		    		for (int i = 0; i < asArrayOfAvailableData (jso).length(); i++){
		    			lstAvailableData.add (asArrayOfAvailableData (jso).get(i));
		    		}
		    		refresh_available_landscapes();
		    		break;
	    		}
		    	case 3:{
		    		lstImputationData.clear();
		    		for (int i = 0; i < asArrayOfSingleLineData (jso).length(); i++){
		    			lstImputationData.add (asArrayOfImputationData (jso).get(i));
		    		}
	    			lsb_viewoptions.setItemSelected(0,true);
	    			refresh_imputation_table("Original Fitness");
	    			break;
	    		}
		    	case 4:{		    	
//		    		Window.alert("queryflag 4 is back");
			    if (asArrayOfSingleLineData (jso).get(0).get_content().substring(0,5) == "Error") {
					lbl_error.setText(asArrayOfSingleLineData (jso).get(0).get_content());
					ErrorPopupPanel.center();
				}else {
		    			String pubemed_link = asArrayOfSingleLineData (jso).get(0).get_content();
		    			Window.open(pubemed_link, "_blank", null);
				}
	    			break;
	    		}
		    	
		    	case 5:{		    		
//		    		Window.alert("JSON: queryflag = 4 is back!");
//		    		String pubemed_link = asArrayOfSingleLineData (jso).get(0).get_content();
//		    		Window.open(pubemed_link, "_blank", null);
	    			break;
	    		}
		    	
		    	
		    	

		    	
		    }
	    	intRequest = 0;
	  }
	  
	  private void refresh_imputation_table(String score_type){
//		  Window.alert("start to refresh fitness landscape......# of records:" + lstImputationData.size());
		  //update legend
  
		  if (score_type ==  "Original Fitness") {
			  panel_right_up_right.clear(); 
			  img_imputation_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_imputation_legend);  			  
		  };
		  if (score_type ==  "Original Fitness - reversed") {
			  panel_right_up_right.clear(); 
			  img_imputation_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_imputation_legend);
		  };
		  if (score_type ==  "Refined Fitness") {
			  panel_right_up_right.clear();
			  img_imputation_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_imputation_legend);
		  };
		  if (score_type ==  "Polyphen Score") {
			  panel_right_up_right.clear();
			  img_polyphen_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_polyphen_legend);
		  };
		  if (score_type ==  "SIFT Score") {
			  panel_right_up_right.clear();
			  img_sift_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_sift_legend);
		  };
		  if (score_type ==  "Provean Score") {
			  panel_right_up_right.clear();
			  img_provean_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_provean_legend);
		  };
		  if (score_type ==  "Allele Frequency") {
			  panel_right_up_right.clear();
			  img_gnomad_legend.setPixelSize((int) Math.round(s_width*(1-panel_left_ratio)*(panel_legend_up_right_ratio)), (int) Math.round(s_height*panel_up_right_ratio));
			  panel_right_up_right.add(img_gnomad_legend);
		  };
		  
	  Double font_factor = 0.8;
	  int largest_font = 16;
	  int default_font = 13;
	  int last_position = 0;

	  final FlexTableWithMouseEvent new_FlexTable_Landscape = new FlexTableWithMouseEvent();
	  final FlexTableWithMouseEvent new_FlexTable_Landscape_fixed = new FlexTableWithMouseEvent();
//	  String[] lst_aa = {"S","A","V","R","D","F","T","I","L","K","G","Y","N","C","P","E","M","W","H","Q","*"};
	  String[] lst_aa = {"A","V","L","I","M","F","Y","W","R","H","K","D","E","S","T","N","Q","G","C","P","*"};
	  String[] lst_sc = {};
	  
	  
      int sc_spanned_cols = 0;
      int pfam_spanned_cols = 0;
      int min_pos = lstImputationData.get(0).get_aa_pos() - lstImputationData.get(0).get_aa_pos_index() +1;
	  //fill the landscape 
	  for (int i = 0; i<lstImputationData.size(); i++){
		  //Window.alert("painting records:" + i);
//		  int cur_aa_pos = lstImputationData.get(i).get_aa_pos();
		  int cur_aa_pos = lstImputationData.get(i).get_aa_pos_index();
		  String cur_aa_alt = lstImputationData.get(i).get_aa_alt();
		  String cur_aa_ref = lstImputationData.get(i).get_aa_ref();
		  String cur_aa_psipred = lstImputationData.get(i).get_aa_psipred();
//		  int cur_ss_end_pos = lstImputationData.get(i).get_ss_end_pos();
		  int cur_ss_end_pos = lstImputationData.get(i).get_ss_end_pos_index();
		  String cur_hmm_id = lstImputationData.get(i).get_hmm_id();
//		  int cur_pfam_end_pos = lstImputationData.get(i).get_pfam_end_pos();
		  int cur_pfam_end_pos = lstImputationData.get(i).get_pfam_end_pos_index();
		  Double cur_fitness_se_refine = lstImputationData.get(i).get_fitness_se_refine();
		  int cur_se_fontsize = default_font;
		  Double cur_fitness = lstImputationData.get(i).get_fitness_refine();
		  String cur_colorcode = lstImputationData.get(i).get_fitness_refine_colorcode();
//		  Double cur_asa_mean = lstImputationData.get(i).get_asa_mean_normalized();
		  String cur_asa_colorcode  = lstImputationData.get(i).get_asa_colorcode();
		  
		  if (score_type == "Original Fitness"){
			  cur_fitness = lstImputationData.get(i).get_fitness_org();
			  cur_colorcode = lstImputationData.get(i).get_fitness_org_colorcode();
			  cur_se_fontsize = lstImputationData.get(i).get_se_org_fontsize();
		  }
		  
		  if (score_type == "Original Fitness - reversed"){
			  cur_fitness = lstImputationData.get(i).get_fitness_reverse();
			  cur_colorcode = lstImputationData.get(i).get_fitness_reverse_colorcode();
			  cur_se_fontsize = lstImputationData.get(i).get_se_org_fontsize();
		  }
		  
		  if (score_type == "Refined Fitness"){
			  cur_fitness = lstImputationData.get(i).get_fitness_refine();
			  cur_colorcode = lstImputationData.get(i).get_fitness_refine_colorcode();
			  cur_se_fontsize = lstImputationData.get(i).get_se_refine_fontsize();
		  }
		  
		  if (score_type == "Polyphen Score"){
			  cur_fitness = lstImputationData.get(i).get_polyphen_score();
			  cur_colorcode = lstImputationData.get(i).get_polyphen_colorcode();
			  cur_fitness_se_refine = null;
		  }
		  
		  if (score_type == "SIFT Score"){
			  cur_fitness = lstImputationData.get(i).get_sift_score();
			  cur_colorcode = lstImputationData.get(i).get_sift_colorcode();
			  cur_fitness_se_refine = null;
		  }
		  
		  if (score_type == "Provean Score"){
			  cur_fitness = lstImputationData.get(i).get_provean_score();
			  cur_colorcode = lstImputationData.get(i).get_provean_colorcode();
			  cur_fitness_se_refine = null;
		  }
		  
		  if (score_type == "Allele Frequency"){
			  cur_fitness = lstImputationData.get(i).get_gnomad_af();
			  cur_colorcode = lstImputationData.get(i).get_gnomad_colorcode();
			  cur_fitness_se_refine = null;
		  }

		  int row_index = -1;
		  int column_index = cur_aa_pos-1;

		  //get row index
		  for (int j = 0; j<lst_aa.length; j++){
			  if (lst_aa[j] == cur_aa_alt){
				  row_index = j+5;
			  }
		  }			  
		  //get last position
		  if (cur_aa_pos > last_position){last_position = cur_aa_pos;}
		  if ((score_type == "Original Fitness") || (score_type == "Refined Fitness")) {
//			  if (cur_se_fontsize == largest_font)
//				  new_FlexTable_Landscape.setText(row_index, column_index, "o");
//			  else
				  new_FlexTable_Landscape.setText(row_index, column_index, "|");
		  }else
			  new_FlexTable_Landscape.setText(row_index, column_index, " ");
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(row_index, column_index).setAttribute("id",i+"");
		  
		  //paint fitness color and 			  
		  if (cur_fitness != null){
			  new_FlexTable_Landscape.getFlexCellFormatter().getElement(row_index, column_index).setAttribute("style","color:" + "#000000" +";"+"font-size:" + cur_se_fontsize*font_factor +"px;" +"background-color:" + cur_colorcode +";border-style: solid;border-color: grey;border-width: 1px;text-align:center");}			  
		  else
			  new_FlexTable_Landscape.getFlexCellFormatter().getElement(row_index, column_index).setAttribute("style","color:" + "#C0C0C0" +";"+"font-size:" + cur_se_fontsize*font_factor +"px;" +"background-color:" + "#C0C0C0" + ";border-style: solid;border-color: grey;border-width: 1px;text-align:center");

		  //Second row of landscape: add aa_Ref
		  new_FlexTable_Landscape.setText(1, column_index, cur_aa_ref);
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(1, column_index).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px");
		  			  
		  //set the row secondary structure
		  if (cur_aa_psipred != null){
//			  Window.alert(cur_aa_psipred +":" + (column_index+1) + ":" + cur_ss_end_pos);
			  new_FlexTable_Landscape.setText(3, column_index-sc_spanned_cols, cur_aa_psipred);
			  new_FlexTable_Landscape.getFlexCellFormatter().setColSpan(3, column_index-sc_spanned_cols, cur_ss_end_pos-column_index);
			  if (cur_aa_psipred == "C"){
				  new_FlexTable_Landscape.getFlexCellFormatter().getElement(3, column_index-sc_spanned_cols).setAttribute("style","color:white;background-color:silver;text-align:center;font-size:" + default_font*font_factor + "px");}
			  if (cur_aa_psipred == "E"){
				  new_FlexTable_Landscape.getFlexCellFormatter().getElement(3, column_index-sc_spanned_cols).setAttribute("style","color:white;background-color:blue;text-align:center;font-size:" + default_font*font_factor + "px");}
			  if (cur_aa_psipred == "H"){
				  new_FlexTable_Landscape.getFlexCellFormatter().getElement(3, column_index-sc_spanned_cols).setAttribute("style","color:white;background-color:red;text-align:center;font-size:" + default_font*font_factor + "px");}
			  sc_spanned_cols = sc_spanned_cols + cur_ss_end_pos-column_index-1;				  
		  }
		  
		  //set the row  pfam
		  if (cur_hmm_id != null){
			  //Window.alert(cur_hmm_id +":" + (column_index+1) + ":" + cur_pfam_end_pos);
			  new_FlexTable_Landscape.setText(4, column_index-pfam_spanned_cols, cur_hmm_id);
			  new_FlexTable_Landscape.getFlexCellFormatter().setColSpan(4, column_index-pfam_spanned_cols, cur_pfam_end_pos-column_index);
			  new_FlexTable_Landscape.getFlexCellFormatter().getElement(4, column_index-pfam_spanned_cols).setAttribute("style","color:black;background-color:yellow;text-align:center;font-size:" + default_font*font_factor + "px");
			  pfam_spanned_cols = pfam_spanned_cols + cur_pfam_end_pos-column_index-1;
		  }
		  
		  //set the row asa
		  new_FlexTable_Landscape.setText(2, column_index, "");
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(2, column_index).setAttribute("style","color:" + "#000000" +";"+"font-size:" + cur_se_fontsize*font_factor +"px;" +"background-color:" + cur_asa_colorcode +";border-style: solid;border-color: grey;border-width: 1px;text-align:center");
	  }
//	  Window.alert("last position:" + last_position);
	  
	  //fisrt row and last row  of landscape: control same width for all cells 
	  for (int j = 0; j<=last_position; j++){
		  //first row of landscape: add position
		  new_FlexTable_Landscape.setText(0, j, "" + (j + min_pos));
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(0, j).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px");
		  
		  //last row of landscape control same width for all cells 
		  new_FlexTable_Landscape.setText(26, j, "*****");
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(26, j).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center;visibility:collapse");
//		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(26, j).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
		  
	  }		  
  
	  //last column of the landscape ; control same height for all cells		  
	  for (int j = 0; j<=lst_aa.length+4; j++){
		  new_FlexTable_Landscape.setText(j, last_position, "I");
		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(j, last_position).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center;visibility:collapse");
//		  new_FlexTable_Landscape.getFlexCellFormatter().getElement(j, last_position).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
	  }		  
	  
	  //first column the landscape (fixed), separate FlexTable		  
	  for (int j = 0; j<=lst_aa.length+4; j++){
		  if (j == 2){
			  new_FlexTable_Landscape_fixed.setText(j, 0, "ASA");
			  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 0).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
		  }
		  if (j == 3){
			  new_FlexTable_Landscape_fixed.setText(j, 0, "SS");
			  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 0).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
		  }
		  if (j == 4){
			  new_FlexTable_Landscape_fixed.setText(j, 0, "Pfam");
			  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 0).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
		  }
		  
		  if (j > 4){
			  new_FlexTable_Landscape_fixed.setText(j, 0, lst_aa[j-5]);
			  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 0).setAttribute("style","text-align:center;font-size:" + default_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
		  }
	  }

	  //last column the landscape (fixed), separate FlexTable, control the same height of all cell 
	  for (int j = 0; j<=lst_aa.length+4; j++){
		  new_FlexTable_Landscape_fixed.setText(j, 1, "I");
		  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 1).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor +"px;border-style: solid;border-color: grey;border-width: 1px;text-align:center;visibility:collapse");
//		  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(j, 1).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor +"px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
	  }
	  
	  //last row the landscape (fixed), separate FlexTable, control the same height of all cell 
	  for (int j = 0; j<=1; j++){
		  new_FlexTable_Landscape_fixed.setText(26, j, "*****");
		  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(26, j).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center;visibility:collapse");
//		  new_FlexTable_Landscape_fixed.getFlexCellFormatter().getElement(26, j).setAttribute("style","text-align:center;font-size:" + largest_font*font_factor + "px;border-style: solid;border-color: grey;border-width: 1px;text-align:center");
	  }
	  
	  new_FlexTable_Landscape.addMouseMoveHandler(new MouseMoveHandler(){
		  public void onMouseMove(MouseMoveEvent event){
			  if (blnFixPopUp == false ){
				  Element td = new_FlexTable_Landscape.getCellForMouseMoveEvent(event);
				  if (td != null){
					  td.getStyle().setProperty("cursor", "pointer");						  
					  String str_id = td.getAttribute("id");
					  LandscapePopHitTable.removeAllRows();
					  int  id = Integer.parseInt(str_id);
					  
					  LandscapePopHitTable.setText(0, 0, "Position: " +lstImputationData.get(id).get_aa_pos() + " [ " + lstImputationData.get(id).get_aa_ref()+" ⇨ " + lstImputationData.get(id).get_aa_alt() + " ]");
					  LandscapePopHitTable.setText(1, 0, "Original Fitness");
					  LandscapePopHitTable.setText(1, 1, lstImputationData.get(id).get_fitness_org() + "");
					  LandscapePopHitTable.setText(2, 0, "Original Fitness SE");
					  LandscapePopHitTable.setText(2, 1, "±" + lstImputationData.get(id).get_fitness_se_org()+"");
					  LandscapePopHitTable.setText(3, 0, "Refined Fitness");
					  LandscapePopHitTable.setText(3, 1, lstImputationData.get(id).get_fitness_refine()+"");
					  LandscapePopHitTable.setText(4, 0, "Refined Fitness SE");
					  LandscapePopHitTable.setText(4, 1, "±" + lstImputationData.get(id).get_fitness_se_refine());
					  LandscapePopHitTable.setText(5, 0, "Quality Score");
					  LandscapePopHitTable.setText(5, 1, lstImputationData.get(id).get_quality_score()+"");
					  LandscapePopHitTable.setText(6, 0, "Polyhen Score");
					  LandscapePopHitTable.setText(6, 1, lstImputationData.get(id).get_polyphen_score()+"");
					  LandscapePopHitTable.setText(7, 0, "SIFT Score");
					  LandscapePopHitTable.setText(7, 1, lstImputationData.get(id).get_sift_score()+"");
					  LandscapePopHitTable.setText(8, 0, "Provean Score");
					  LandscapePopHitTable.setText(8, 1, lstImputationData.get(id).get_provean_score()+"");	
					  LandscapePopHitTable.setText(9, 0, "Blosum62");
					  LandscapePopHitTable.setText(9, 1, lstImputationData.get(id).get_blosum62()+"");
					  LandscapePopHitTable.setText(10, 0, "Allel Frequency");
					  LandscapePopHitTable.setText(10, 1, lstImputationData.get(id).get_gnomad_af()+"");
					  LandscapePopHitTable.setText(11, 0, "Clinvar");
					  LandscapePopHitTable.setText(11, 1, "N/A");
					  
					  LandscapePopHitTable.getFlexCellFormatter().setColSpan(0, 0, 2);

					  for (int i = 1; i<=11; i++){
						  LandscapePopHitTable.getFlexCellFormatter().setStyleName(i, 0, "popup_table_cellformat");
						  LandscapePopHitTable.getFlexCellFormatter().setStyleName(i, 1, "popup_table_cellformat");
//						  LandscapePopHitTable.getFlexCellFormatter().getElement(i, 0).setAttribute("style", "background-color:silver; border-style:solid; border-color:grey; border-width:1px;");
					  }
					  LandscapePopHitTable.getFlexCellFormatter().setStyleName(0, 0, "popup_table_headercellformat");
					  
					  final int y = td.getAbsoluteTop();
					  final int x = td.getAbsoluteLeft();
					  ChartPopupPanel.setPopupPositionAndShow(new PopupPanel.PositionCallback() { 
                          public void setPosition(int offsetWidth, int offsetHeight) { 
                        	  if  (y - offsetHeight < 0)
                        		  ChartPopupPanel.setPopupPosition(x-offsetWidth,y);
                        	  else
                        		  ChartPopupPanel.setPopupPosition(x-offsetWidth,y-offsetHeight);                            	
    						  blnPopUp = true;
                          } 
					  }); 
				  }
			  }
		  }
	  });
	  
	  new_FlexTable_Landscape.addMouseOutHandler(new MouseOutHandler(){
		  public void onMouseOut(MouseOutEvent event){
			  if (blnFixPopUp == false){
				  ChartPopupPanel.hide();
			  	  blnPopUp = false;
		  	  }
		  }
	  });
	  
	  img_loading.setVisible(false);
	  panel_down_right.clear();
	  panel_down_right.addWest(new_FlexTable_Landscape_fixed, 50);
	  panel_down_right.add(panel_down_right_scroll);
	  panel_down_right_scroll.clear();
	  panel_down_right_scroll.add(new_FlexTable_Landscape);
	  panel_down_right.forceLayout();
  
	  btn_download_csv.setEnabled(true);
	  btn_download_figure.setEnabled(true);
//	  Window.alert(session_id);
	  if (session_id.substring(0, 1) == "*") {
		  btn_pubmed_link.setEnabled(true);}
	  else {
		  btn_pubmed_link.setEnabled(false);}
	  }
 
}