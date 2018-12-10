#pragma once
#include "CUDA_Spice.h"

namespace GUI {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::Button^  button1;
	private: System::Windows::Forms::Label^  inLabel;
	private: System::Windows::Forms::TextBox^  inputTB;
	private: System::Windows::Forms::TextBox^  logBox;




	private: System::Windows::Forms::Label^  label2;
	private: System::Windows::Forms::Button^  startButton;
	private: System::Windows::Forms::CheckBox^  opCB;
	private: System::Windows::Forms::CheckBox^  tranCB;


	private: System::Windows::Forms::CheckBox^  dcSweepCB;

	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  startLabel;
	private: System::Windows::Forms::Label^  stopLabel;
	private: System::Windows::Forms::Label^  stepLabel;



	private: System::Windows::Forms::TextBox^  startTB;
	private: System::Windows::Forms::TextBox^  stopTB;
	private: System::Windows::Forms::TextBox^  stepTB;
	private: System::Windows::Forms::TextBox^  outputTB;

	private: System::Windows::Forms::Label^  outLabel;
	private: System::Windows::Forms::Label^  elemLabel;
	private: System::Windows::Forms::TextBox^  elemTB;

	private: CUDA_Spice^ CSpi = (gcnew CUDA_Spice());



	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->inLabel = (gcnew System::Windows::Forms::Label());
			this->inputTB = (gcnew System::Windows::Forms::TextBox());
			this->logBox = (gcnew System::Windows::Forms::TextBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->startButton = (gcnew System::Windows::Forms::Button());
			this->opCB = (gcnew System::Windows::Forms::CheckBox());
			this->tranCB = (gcnew System::Windows::Forms::CheckBox());
			this->dcSweepCB = (gcnew System::Windows::Forms::CheckBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->startLabel = (gcnew System::Windows::Forms::Label());
			this->stopLabel = (gcnew System::Windows::Forms::Label());
			this->stepLabel = (gcnew System::Windows::Forms::Label());
			this->startTB = (gcnew System::Windows::Forms::TextBox());
			this->stopTB = (gcnew System::Windows::Forms::TextBox());
			this->stepTB = (gcnew System::Windows::Forms::TextBox());
			this->outputTB = (gcnew System::Windows::Forms::TextBox());
			this->outLabel = (gcnew System::Windows::Forms::Label());
			this->elemLabel = (gcnew System::Windows::Forms::Label());
			this->elemTB = (gcnew System::Windows::Forms::TextBox());
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(12, 12);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(130, 34);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Open File";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// inLabel
			// 
			this->inLabel->AutoSize = true;
			this->inLabel->Location = System::Drawing::Point(12, 65);
			this->inLabel->Name = L"inLabel";
			this->inLabel->Size = System::Drawing::Size(85, 17);
			this->inLabel->TabIndex = 1;
			this->inLabel->Text = L"Current File:";
			// 
			// inputTB
			// 
			this->inputTB->BackColor = System::Drawing::SystemColors::ControlLightLight;
			this->inputTB->Location = System::Drawing::Point(110, 65);
			this->inputTB->MaximumSize = System::Drawing::Size(450, 50);
			this->inputTB->MinimumSize = System::Drawing::Size(450, 50);
			this->inputTB->Multiline = true;
			this->inputTB->Name = L"inputTB";
			this->inputTB->ReadOnly = true;
			this->inputTB->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->inputTB->Size = System::Drawing::Size(450, 50);
			this->inputTB->TabIndex = 0;
			this->inputTB->Text = L"None";
			// 
			// logBox
			// 
			this->logBox->BackColor = System::Drawing::SystemColors::ControlLightLight;
			this->logBox->Cursor = System::Windows::Forms::Cursors::IBeam;
			this->logBox->Location = System::Drawing::Point(15, 325);
			this->logBox->MaximumSize = System::Drawing::Size(550, 100);
			this->logBox->MinimumSize = System::Drawing::Size(550, 100);
			this->logBox->Multiline = true;
			this->logBox->Name = L"logBox";
			this->logBox->ReadOnly = true;
			this->logBox->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->logBox->Size = System::Drawing::Size(550, 100);
			this->logBox->TabIndex = 2;
			this->logBox->Text = L"----- Log -----";
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(12, 120);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(84, 17);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Simulations:";
			// 
			// startButton
			// 
			this->startButton->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(128)), static_cast<System::Int32>(static_cast<System::Byte>(255)),
				static_cast<System::Int32>(static_cast<System::Byte>(128)));
			this->startButton->FlatStyle = System::Windows::Forms::FlatStyle::Popup;
			this->startButton->Font = (gcnew System::Drawing::Font(L"Arial", 16.2F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->startButton->Location = System::Drawing::Point(374, 152);
			this->startButton->Name = L"startButton";
			this->startButton->Size = System::Drawing::Size(191, 93);
			this->startButton->TabIndex = 4;
			this->startButton->Text = L"Start";
			this->startButton->UseVisualStyleBackColor = false;
			this->startButton->Click += gcnew System::EventHandler(this, &MyForm::startButton_Click);
			// 
			// opCB
			// 
			this->opCB->AutoSize = true;
			this->opCB->Location = System::Drawing::Point(27, 140);
			this->opCB->Name = L"opCB";
			this->opCB->Size = System::Drawing::Size(129, 21);
			this->opCB->TabIndex = 5;
			this->opCB->Text = L"Operating Point";
			this->opCB->UseVisualStyleBackColor = true;
			// 
			// tranCB
			// 
			this->tranCB->AutoSize = true;
			this->tranCB->Location = System::Drawing::Point(27, 194);
			this->tranCB->Name = L"tranCB";
			this->tranCB->Size = System::Drawing::Size(90, 21);
			this->tranCB->TabIndex = 6;
			this->tranCB->Text = L"Transient";
			this->tranCB->UseVisualStyleBackColor = true;
			// 
			// dcSweepCB
			// 
			this->dcSweepCB->AutoSize = true;
			this->dcSweepCB->Location = System::Drawing::Point(27, 167);
			this->dcSweepCB->Name = L"dcSweepCB";
			this->dcSweepCB->Size = System::Drawing::Size(95, 21);
			this->dcSweepCB->TabIndex = 7;
			this->dcSweepCB->Text = L"DC Sweep";
			this->dcSweepCB->UseVisualStyleBackColor = true;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(199, 120);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(131, 17);
			this->label3->TabIndex = 8;
			this->label3->Text = L"Sweep Parameters:";
			// 
			// startLabel
			// 
			this->startLabel->AutoSize = true;
			this->startLabel->Location = System::Drawing::Point(200, 170);
			this->startLabel->Name = L"startLabel";
			this->startLabel->Size = System::Drawing::Size(42, 17);
			this->startLabel->TabIndex = 9;
			this->startLabel->Text = L"Start:";
			// 
			// stopLabel
			// 
			this->stopLabel->AutoSize = true;
			this->stopLabel->Location = System::Drawing::Point(200, 198);
			this->stopLabel->Name = L"stopLabel";
			this->stopLabel->Size = System::Drawing::Size(41, 17);
			this->stopLabel->TabIndex = 10;
			this->stopLabel->Text = L"Stop:";
			// 
			// stepLabel
			// 
			this->stepLabel->AutoSize = true;
			this->stepLabel->Location = System::Drawing::Point(200, 226);
			this->stepLabel->Name = L"stepLabel";
			this->stepLabel->Size = System::Drawing::Size(41, 17);
			this->stepLabel->TabIndex = 11;
			this->stepLabel->Text = L"Step:";
			// 
			// startTB
			// 
			this->startTB->Location = System::Drawing::Point(247, 167);
			this->startTB->MaximumSize = System::Drawing::Size(100, 22);
			this->startTB->MinimumSize = System::Drawing::Size(100, 22);
			this->startTB->Name = L"startTB";
			this->startTB->Size = System::Drawing::Size(100, 22);
			this->startTB->TabIndex = 12;
			// 
			// stopTB
			// 
			this->stopTB->Location = System::Drawing::Point(247, 195);
			this->stopTB->MaximumSize = System::Drawing::Size(100, 22);
			this->stopTB->MinimumSize = System::Drawing::Size(100, 22);
			this->stopTB->Name = L"stopTB";
			this->stopTB->Size = System::Drawing::Size(100, 22);
			this->stopTB->TabIndex = 13;
			// 
			// stepTB
			// 
			this->stepTB->Location = System::Drawing::Point(247, 223);
			this->stepTB->MaximumSize = System::Drawing::Size(100, 22);
			this->stepTB->MinimumSize = System::Drawing::Size(100, 22);
			this->stepTB->Name = L"stepTB";
			this->stepTB->Size = System::Drawing::Size(100, 22);
			this->stepTB->TabIndex = 14;
			// 
			// outputTB
			// 
			this->outputTB->Location = System::Drawing::Point(115, 269);
			this->outputTB->MaximumSize = System::Drawing::Size(450, 50);
			this->outputTB->MinimumSize = System::Drawing::Size(450, 50);
			this->outputTB->Multiline = true;
			this->outputTB->Name = L"outputTB";
			this->outputTB->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->outputTB->Size = System::Drawing::Size(450, 50);
			this->outputTB->TabIndex = 15;
			// 
			// outLabel
			// 
			this->outLabel->AutoSize = true;
			this->outLabel->Location = System::Drawing::Point(12, 269);
			this->outLabel->Name = L"outLabel";
			this->outLabel->Size = System::Drawing::Size(81, 17);
			this->outLabel->TabIndex = 16;
			this->outLabel->Text = L"Output File:";
			// 
			// elemLabel
			// 
			this->elemLabel->AutoSize = true;
			this->elemLabel->Location = System::Drawing::Point(200, 142);
			this->elemLabel->Name = L"elemLabel";
			this->elemLabel->Size = System::Drawing::Size(49, 17);
			this->elemLabel->TabIndex = 17;
			this->elemLabel->Text = L"Name:";
			// 
			// elemTB
			// 
			this->elemTB->Location = System::Drawing::Point(247, 139);
			this->elemTB->MaximumSize = System::Drawing::Size(100, 22);
			this->elemTB->MinimumSize = System::Drawing::Size(100, 22);
			this->elemTB->Name = L"elemTB";
			this->elemTB->Size = System::Drawing::Size(100, 22);
			this->elemTB->TabIndex = 18;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(582, 433);
			this->Controls->Add(this->elemTB);
			this->Controls->Add(this->elemLabel);
			this->Controls->Add(this->outLabel);
			this->Controls->Add(this->outputTB);
			this->Controls->Add(this->stepTB);
			this->Controls->Add(this->stopTB);
			this->Controls->Add(this->startTB);
			this->Controls->Add(this->stepLabel);
			this->Controls->Add(this->stopLabel);
			this->Controls->Add(this->startLabel);
			this->Controls->Add(this->label3);
			this->Controls->Add(this->dcSweepCB);
			this->Controls->Add(this->tranCB);
			this->Controls->Add(this->opCB);
			this->Controls->Add(this->startButton);
			this->Controls->Add(this->label2);
			this->Controls->Add(this->logBox);
			this->Controls->Add(this->inputTB);
			this->Controls->Add(this->inLabel);
			this->Controls->Add(this->button1);
			this->Name = L"MyForm";
			this->Text = L"CUDA Circuit Simulator";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion


	private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {

	}
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {

		OpenFileDialog^ openFileDialog1 = gcnew OpenFileDialog;

		openFileDialog1->InitialDirectory = "./";
		openFileDialog1->RestoreDirectory = true;

		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			inputTB->Text = openFileDialog1->FileName;
		}
		logBox->AppendText(Environment::NewLine);
		logBox->AppendText("Loading Netlist...");
		logBox->AppendText(Environment::NewLine);

		String^ path = inputTB->Text;
		int check = CSpi->genNetlists(path);
		if (check != 0) {
			logBox->AppendText("Failed! Bad Netlist?");
			logBox->AppendText(Environment::NewLine);
		}
		else {
			logBox->AppendText("Success!");
			logBox->AppendText(Environment::NewLine);
		}
		

	}
	private: System::Void startButton_Click(System::Object^  sender, System::EventArgs^  e) {
		if (opCB->Checked) {
			logBox->AppendText("Performing OP Sim...");
			logBox->AppendText(Environment::NewLine);

			CSpi->guiOP();

			logBox->AppendText("Writing data output...");
			logBox->AppendText(Environment::NewLine);

			String^ s = outputTB->Text;

			CSpi->outToCSV(s);

			logBox->AppendText("Data written to ");
			logBox->AppendText(s);
			logBox->AppendText(Environment::NewLine);
		}


	}
	};
}
