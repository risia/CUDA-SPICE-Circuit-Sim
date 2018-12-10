#include "MyForm.h"

using namespace GUI;
using namespace System;
using namespace System::Windows::Forms;

[STAThread]
int main(cli::array<System::String ^> ^args) {

	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Application::Run(gcnew MyForm());
	return 0;
}