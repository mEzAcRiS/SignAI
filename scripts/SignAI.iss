; ============================================================
; SignAI - script de instalador (Inno Setup 6)
; Compilar:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\SignAI.iss
; Salida:     dist\Setup_SignAI_1.0.exe
; Requiere:   dist\SignAI\ ya construido con scripts\build_exe.bat
; ============================================================

#define MyAppName "SignAI"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Equipo 4 - Fundamentos de Inteligencia Artificial"
#define MyAppURL "https://github.com/mEzAcRiS/SignAI"
#define MyAppExeName "SignAI.exe"

[Setup]
AppId={{1215F0A2-A025-4D15-824A-BF7A7FBB4CC3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=Setup_{#MyAppName}_{#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#MyAppName}
VersionInfoVersion={#MyAppVersion}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\SignAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; dataset de verificacion para el atajo "verificar CSV" (2.5 MB)
Source: "..\data\hand_landmarks.csv"; DestDir: "{app}\data"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{#MyAppName} - imagen de prueba"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--demo-image ""_internal\data\samples\hand_test.jpg"""; WorkingDir: "{app}"
Name: "{group}\{#MyAppName} - verificar CSV"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--csv ""data\hand_landmarks.csv"""; WorkingDir: "{app}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{autodesktop}\{#MyAppName} - imagen de prueba"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--demo-image ""_internal\data\samples\hand_test.jpg"""; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{autodesktop}\{#MyAppName} - verificar CSV"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--csv ""data\hand_landmarks.csv"""; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
