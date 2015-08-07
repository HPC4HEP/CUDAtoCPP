/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/jul/15
 *      Author: latzori
 */

#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

//TODO this is mandatory?
static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter *R) : ASTConsumer(), CI(comp), Rew(R){ }
	virtual ~MyASTConsumer() { }

	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();
		PP = &CI->getPreprocessor();
		//Rew.setSourceMgr(*SM, *LO);
		//PP->addPPCallbacks(new RewriteIncludesCallback(this));
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {

		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		//SourceLocation loc = firstDecl->getLocation();
		SourceLocation sloc = SM->getSpellingLoc(firstDecl->getLocation());

		//std::cout << "loc " << loc.printToString(*SM) << "\n";
		//std::cout << "sloc " << sloc.printToString(*SM) << "\n";

		//Don't use extern in the include files!!!
//		std::cout << "loc " << loc.printToString(*SM) <<" (filename: " << SM->getFilename(loc).str() << ")\n";
//		std::cout << "sloc " << sloc.printToString(*SM) <<" (filename: " << SM->getFilename(sloc).str() << ")\n";
		//SM->getFileID(loc) != SM->getMainFileID() &&
//		std::cout << "MAIN ID " << SM->getMainFileID().getHashValue();
//		std::cout << " our ID " << SM->getFileID(loc).getHashValue();
//		std::cout << " our s ID " << SM->getFileID(sloc).getHashValue() << "\n";

		// Checking which file we are scanning (same as isFromMainFile property, now deprecated???)
		if( SM->getFileID(sloc) != SM->getMainFileID()){
//			std::cout << "FileID mismatch (Skipped rewrite)\n";
			return true; //Just skip the loc (TODO skip the file?) but continue parsing
		}

		std::cout << "NEW DECL: " << sloc.printToString(*SM) << "\n";

		//FROM CU2CL
        //Walk declarations in group and rewrite
        for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
//        	std::cout << "DEBUG: Analyzing decl " << i << "\n"; //TODO need integer index
            if (DeclContext *dc = dyn_cast<DeclContext>(*i)) {
            	std::cout << "DEBUG: DeclContext dc \n";
                //Basically only handles C++ member functions
                for (DeclContext::decl_iterator di = dc->decls_begin(), de = dc->decls_end(); di != de; ++di) {
                    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*di)) {
                    	std::cout << "DEBUG: FunctionDecl fd inside dc\n";
                        //prevent implicitly defined functions from being rewritten
                    	// (since there's no source to rewrite..)
                        if (!fd->isImplicit()) {
                        	std::cout << "DEBUG: fd->isImplicit = false (Action required)\n";
                            RewriteHostFunction(fd);
//                            RemoveFunction(fd, KernelRewrite);
//                        	if (fd->getNameAsString() == MainFuncName) {
//                                RewriteMain(fd);
//                            }
                        } else {
                            std::cout << "DEBUG: fd->isImplicit = true (Skipped rewrite)\n";
                        }
                    }
                }
            }
            //Handles globally defined C or C++ functions
            if (FunctionDecl *fd = dyn_cast<FunctionDecl>(*i)) {
            	std::cout << "DEBUG: FunctionDecl fd\n";
            	//Don't translate explicit template specializations
                if(fd->getTemplatedKind() == clang::FunctionDecl::TK_NonTemplate || fd->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate) {
                    if (fd->hasAttr<CUDAGlobalAttr>() || fd->hasAttr<CUDADeviceAttr>()) {
                    	std::cout << "DEBUG: Global and/or Device attrs (Action required)\n";
                    	//Device function, so rewrite kernel
//                        RewriteKernelFunction(fd);
                        if (fd->hasAttr<CUDAHostAttr>()){
                        	std::cout << "DEBUG: Also Host attr (Action required)\n";
                            //Also a host function, so rewrite host
//                            RewriteHostFunction(fd);
                        } else {
                        	std::cout << "DEBUG: Only device, remove from host?\n";
                            //Simply a device function, so remove from host
//                            RemoveFunction(fd, HostRewrite);
                        }
                    } else {
                    	std::cout << "DEBUG: Only host function (Action required)\n";
                        //Simply a host function, so rewrite
                        RewriteHostFunction(fd);
//                        if (CUDAHostAttr *attr = fd->getAttr<CUDAHostAttr>()) {
//                        	std::cout << "DEBUG: Found CUDAHostAttr...";
//                            SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
//                            std::cout << " at loc " << instLoc.printToString(*SM);
//                            SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
//                            std::cout << " ending at " << PP->getLocForEndOfToken(instLoc).printToString(*SM);
//                            Rew->ReplaceText(instLoc, Rew->getRangeSize(realRange), "");
//                            std::cout << " Removed!\n";
//                        }
                        //and remove from kernel
//                        RemoveFunction(fd, KernelRewrite);

//                        if (fd->getNameAsString() == MainFuncName) {
//                            RewriteMain(fd);
//                        }
                    }
                } else {
                    if (fd->getTemplateSpecializationInfo())
                    	std::cout << "DEBUG: fd->getTemplateSpecializationInfo = true (Skip?)\n";
                    else
                    	std::cout << "DEBUG: Non-rewriteable function without TemplateSpecializationInfo detected?\n";
                }
            } else if (VarDecl *vd = dyn_cast<VarDecl>(*i)) {
            	std::cout << "DEBUG: VarDecl vd\n";
//                RemoveVar(vd, KernelRewrite);
//                RewriteHostVarDecl(vd);
            //Rewrite Structs here
            //Ideally, we should keep the expression inside parentheses ie __align__(<keep this>)
            // and just wrap it with __attribute__((aligned (<kept Expr>)))
            //TODO: Finish struct attribute translation
        	} else if (RecordDecl * rd = dyn_cast<RecordDecl>(*i)) {
        		std::cout << "DEBUG: RecordDecl rd\n";
                if (rd->hasAttrs()) {
                	std::cout << "DEBUG: rd->hasAttrs = true (action required)\n";
                    for (Decl::attr_iterator at = rd->attr_begin(), at_e = rd->attr_end(); at != at_e; ++at) {
                        if (AlignedAttr *align = dyn_cast<AlignedAttr>(*at)) {
                            if (!align->isAlignmentDependent()) {
                                llvm::errs() << "Found an aligned struct of size: " << align->getAlignment(rd->getASTContext()) << " (bits)\n";
                            } else {
                                llvm::errs() << "Found a dependent alignment expresssion\n";
                            }
                        } else {
                            llvm::errs() << "Found other attrib\n";
                        }
                    }
                }
            }
            //TODO rewrite type declarations
        }
        return true;
}

private:
    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
    Rewriter *Rew;

    //Simple function to strip attributes from host functions that may be declared as
    // both __host__ and __device__, then passes off to the host-side statement rewriter
    void RewriteHostFunction(FunctionDecl *hostFunc) {
    	//Remove any CUDA function attributes
        if (CUDAHostAttr *attr = hostFunc->getAttr<CUDAHostAttr>()) {
        	std::cout << "DEBUG: HostAttr rewriting attempt\n";
            RewriteAttr(attr, "", *Rew);
        }
        if (CUDADeviceAttr *attr = hostFunc->getAttr<CUDADeviceAttr>()) {
            RewriteAttr(attr, "", *Rew);
        }

        //Rewrite the body
        if (Stmt *body = hostFunc->getBody()) {
            RewriteHostStmt(body);
        }
        //CurVarDeclGroups.clear();
    }

    void RewriteHostStmt(Stmt *s) {}
    bool RewriteHostExpr(Expr *e, std::string &newExpr) {}

    //The workhorse that takes the constructed replacement attribute and inserts it in place of the old one
    void RewriteAttr(Attr *attr, std::string replace, Rewriter &rewrite){
    	std::cout << "DEBUG: RewriteAttr\n";
        SourceLocation instLoc = SM->getExpansionLoc(attr->getLocation());
        SourceRange realRange(instLoc, PP->getLocForEndOfToken(instLoc));
        rewrite.ReplaceText(instLoc, rewrite.getRangeSize(realRange), replace);
    }
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, &TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
